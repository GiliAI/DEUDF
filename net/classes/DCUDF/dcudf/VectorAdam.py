import torch

class VectorAdam(torch.optim.Optimizer):

    # 初始化优化器参数，lr,betas,eps/axis.
    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-8, axis=-1):
        defaults = dict(lr=lr, betas=betas, eps=eps, axis=axis)
        super(VectorAdam, self).__init__(params, defaults)

    # 实现了参数更新，对于每个参数组，
    # param_group中的参数来更新，使用梯度，grad和学习率lr,根据Adam算法步骤更新参数。
    def __setstate__(self, state):
        super(VectorAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, N = None, loss = None, selected = None):
        '''
        实现了一种优化算法，用于在训练过程中更新参数以最小化损失函数。它基于动量方法和自适应学习率的思想，通过计算梯度的一阶和二阶矩估计来调整参数的更新幅度。
        ---
            1、遍历每个参数组(param_group):对于每个参数组,获取学习率lr、动量参数b1 b2、缩小项eps,以及可选的轴,axis
            ---
            2、对于每个参数p:
                -检查并进行懒惰初始化：如果当前参数的状态 state 为空，则初始化为零张量。
                -更新步数：将状态中的步数 step 加1。
                -计算梯度 g1:将之前的梯度 g1 乘以动量参数 b1,并加上当前参数的梯度 grad 乘以 (1-b1)。
                -如果指定了轴 axis:
                    -计算梯度平方的范数：在指定轴上计算梯度的范数，并扩展维度后复制到与梯度相同的维度。
                    -计算梯度平方 g2:将之前的梯度平方 g2 乘以动量参数 b2,并加上计算得到的梯度平方 grad_sq 乘以 (1-b2)。
                -否则：
                    -计算梯度平方 g2:将之前的梯度平方 g2 乘以动量参数 b2,并加上当前参数的梯度平方 grad.square() 乘以 (1-b2)。
                -计算 m1 和 m2:通过除以衰减系数的差值来校正梯度平均值 g1 和 g2。
                -计算最终的梯度：将校正后的 g1 和 g2 进行处理，并除以 eps 加上 g2 开方的结果。
                -更新参数：将参数 p 的值减去计算得到的梯度 gr 乘以学习率 lr。
        '''
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            axis = group['axis']
            for p in group["params"]:
                state = self.state[p]
                # Lazy initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)

                g1 = state["g1"]
                g2 = state["g2"]
                state["step"] += 1
                grad = p.grad.data
                # print(f"grad:{grad.shape}")

                g1.mul_(b1).add_(grad, alpha=1-b1)
                if axis is not None:
                    dim = grad.shape[axis]
                    grad_norm = torch.norm(grad, dim=axis).unsqueeze(axis).repeat_interleave(dim, dim=axis)
                    grad_sq = grad_norm * grad_norm
                    g2.mul_(b2).add_(grad_sq, alpha=1-b2)
                else:
                    g2.mul_(b2).add_(grad.square(), alpha=1-b2)

                m1 = g1 / (1-(b1**state["step"]))
                m2 = g2 / (1-(b2**state["step"]))
                gr = m1 / (eps + m2.sqrt())
                # 确保 gr 和 p 的形状匹配
                # gr = gr[selected]
                # print(gr.shape)
                if N is not None:
                    # 投影似乎并不会有很好的效果
                    # n_len_sq = torch.sum(N ** 2, axis=-1, keepdim=True)
                    # proj_gr = torch.sum(gr * N, axis=-1, keepdim=True) * N

                    # # Update parameter along the projected gradient
                    # p.data.sub_(proj_gr, alpha=lr)

                    loss = loss.unsqueeze(1)

                    gr_length = torch.norm(gr)
                    N_normalized = N / torch.norm(N)# 将向量 N 归一化为单位向量                    
                    N = N_normalized * gr_length# 调整向量 N 的长度，使其与向量 gr 相同
                    # N = N[selected].float()
                    N = loss * N + (1 - loss) * gr
                    # gr_normalized = torch.mul(gr_length.unsqueeze(-1), N)
                    # p.data[selected].sub_(N, alpha=lr)
                    # p.data[selected] = p.data[selected].sub(N, alpha=lr)
                    p.data.sub_(N,alpha=lr)
                else:
                    # Update parameter using regular gradient
                    p.data.sub_(gr, alpha=lr)
