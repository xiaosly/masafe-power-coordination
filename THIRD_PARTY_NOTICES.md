# Third-party notices

## MACPO / MAPPO-L

`mappo_lagrangian/` and the training components in `mappo_lag/core/` are adapted from the MAPPO-L implementation of
[Multi-Agent Constrained Policy Optimisation](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation)
(MACPO), released under the MIT License (copyright lines of its license file: "Copyright (c) 2021 anybodyany"
and "Copyright (c) 2020 Tianshou contributors"), restructured for the power-system cases.

MACPO builds on [MAPPO](https://github.com/marlbenchmark/on-policy) (MIT License, Copyright (c) 2021 MAPPO
contributors), whose vectorized environments are in turn adapted from
[OpenAI Baselines](https://github.com/openai/baselines) (MIT License, Copyright (c) 2017 OpenAI).

## pandapower

The IEEE 33-bus and 30-bus test systems are loaded from [pandapower](https://www.pandapower.org)
(BSD 3-Clause License).

## MIT License (applies to the components above)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
