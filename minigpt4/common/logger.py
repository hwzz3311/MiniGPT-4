"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import logging
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist

from minigpt4.common import dist_utils


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    用于跟踪一系列的值，并提供对于滑动窗口或全局数据平均值的访问
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        # 队列的用法，有点意思，涨知识了
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        在多进程训练时用于同步计数和总和。这个函数使用分布式计算工具（dist_utils 和 dist）来同步计数和总和，以确保不同进程之间的计算一致性。
        """
        # 检查是否存在可用且已初始化的分布式环境
        if not dist_utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        # 用于实现分布式环境中的同步操作。在分布式训练中，不同的进程在执行时可能会有不同的速度，为了确保所有进程都完成了特定的操作之后再继续，可以使用 dist.barrier() 进行同步。
        # 在调用这个函数后，所有进程将被阻塞，直到所有进程都到达了这个同步点。
        dist.barrier()
        # 这个函数执行全局归约操作。在分布式训练中，可能需要在所有进程中对某个值进行归约操作，例如求和、平均等。
        # dist.all_reduce(t) 的作用是将所有进程的 t 值相加并在所有进程中广播结果，确保所有进程都具有相同的值。
        dist.all_reduce(t)
        # 取出所有进程的count和total的和
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f}".format(name, meter.global_avg))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """在每次迭代中记录数据处理时间和迭代时间，然后根据 print_freq 的值，定期打印出一条信息来报告迭代进展和指标数据。
            最后，在迭代结束时，会打印出总共花费的时间以及每次迭代平均花费的时间。
        Args:
            iterable (_type_): _description_
            print_freq (_type_): _description_
            header (_type_, optional): _description_. Defaults to None.

        Yields:
            _type_: _description_
        """
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        # 需要打印的log 格式
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        # 统计显存
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            # 使用 yield 关键字会将函数转化为一个生成器（generator）
            yield obj
            iter_time.update(time.time() - end)
            # 判断是否打印 print_freq ，打印步数
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def setup_logger():
    # 只有主进程的等级为info、其他进程的等级为warn
    logging.basicConfig(
        level=logging.INFO if dist_utils.is_main_process() else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
