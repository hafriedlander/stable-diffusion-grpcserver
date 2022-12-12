import torch
from tqdm import trange


@torch.no_grad()
def sample_dpmpp_2m(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    warmup_lms=False,
    ddim_cutoff=0.0,
):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None and warmup_lms:
            r = 1 / 2
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_i = model(x_2, sigma_fn(s) * s_in, **extra_args)
        elif sigmas[i + 1] <= ddim_cutoff or old_denoised is None:
            denoised_i = denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_i = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
        x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_i
        old_denoised = denoised
    return x
