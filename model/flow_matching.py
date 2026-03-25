import torch

class FlowMatching:
    def __init__(self, img_size=256, img_channels=3, device="cpu"):
        self.img_size = img_size
        self.img_channels = img_channels
        self.device = device

    def sample_times(self, n):
        # continuous t in [0, 1]
        return torch.rand(n, device=self.device)

    def make_xt_and_target(self, x1, t):
        # x0 ~ N(0, I)
        x0 = torch.randn_like(x1)
        t_ = t.view(-1, 1, 1, 1)
        xt = (1 - t_) * x0 + t_ * x1
        v_target = x1 - x0
        return xt, v_target, x0

    @torch.no_grad()
    def sample(
        self,
        model,
        condition=None,
        n=None,
        steps=50,
        x_init=None,
        to_uint8=True,
        method="heun",   # "euler" or "heun"
        ):
        """
        Rectified Flow ODE sampling.

        method:
            - "euler": first-order Euler
            - "heun": second-order Heun / RK2 (recommended)
        """
        model.eval()

        # infer batch size
        if n is None:
            if condition is None:
                raise ValueError("n must be provided when condition is None")
            if isinstance(condition, dict):
                n = next(iter(condition.values())).shape[0]
            else:
                n = condition.shape[0]

        # initial noise (x_0)
        if x_init is None:
            x = torch.randn(
                (n, self.img_channels, self.img_size, self.img_size),
                device=self.device,
            )
        else:
            x = x_init.clone()

        # time grid
        ts = torch.linspace(0.0, 1.0, steps + 1, device=self.device)

        for i in range(steps):
            t = torch.full((n,), ts[i].item(), device=self.device)
            t_next = torch.full((n,), ts[i + 1].item(), device=self.device)
            dt = ts[i + 1] - ts[i]

            # ---- Euler step ----
            v1 = model(x, t, condition)

            if method == "euler":
                x = x + dt * v1
                continue

            # ---- Heun / RK2 ----
            x_euler = x + dt * v1
            v2 = model(x_euler, t_next, condition)

            x = x + dt * 0.5 * (v1 + v2)

        model.train()

        if to_uint8:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)

        return x