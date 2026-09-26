import numpy as np

from cnn_network import Conv2D, Dense


def relative_error(analytic, numeric):
    denom = max(np.linalg.norm(analytic), np.linalg.norm(numeric), 1e-8)
    return np.linalg.norm(analytic - numeric) / denom


def numerical_gradient(param, compute_loss, epsilon=1e-5):
    """Central-difference gradient of compute_loss() w.r.t. every entry of param."""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]

        param[idx] = original + epsilon
        loss_plus = compute_loss()

        param[idx] = original - epsilon
        loss_minus = compute_loss()

        param[idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        it.iternext()
    return grad


def check_conv2d():
    np.random.seed(0)
    batch, in_channels, out_channels = 2, 2, 3
    height, width, kernel_size = 5, 5, 3

    layer = Conv2D(in_channels, out_channels, kernel_size)
    x = np.random.randn(batch, in_channels, height, width)

    out = layer.forward(x)
    upstream = np.random.randn(*out.shape)  # fixed random projection direction

    def compute_loss():
        output = layer.forward(x)
        return np.sum(output * upstream)

    layer.forward(x)
    d_x_analytic = layer.backward(upstream)
    d_W_analytic = layer.d_W
    d_b_analytic = layer.d_b

    # d_W/d_b are averaged over batch by convention (matching mlp_network's
    # style); d_x is not, so only the parameter grads need rescaling here.
    d_W_numeric = numerical_gradient(layer.W, compute_loss) / batch
    d_b_numeric = numerical_gradient(layer.b, compute_loss) / batch
    d_x_numeric = numerical_gradient(x, compute_loss)

    print("Conv2D d_W relative error:", relative_error(d_W_analytic, d_W_numeric))
    print("Conv2D d_b relative error:", relative_error(d_b_analytic, d_b_numeric))
    print("Conv2D d_x relative error:", relative_error(d_x_analytic, d_x_numeric))


def check_dense():
    np.random.seed(0)
    in_features, out_features, batch = 6, 4, 3

    layer = Dense(in_features, out_features)
    x = np.random.randn(in_features, batch)

    out = layer.forward(x)
    upstream = np.random.randn(*out.shape)

    def compute_loss():
        output = layer.forward(x)
        return np.sum(output * upstream)

    layer.forward(x)
    d_x_analytic = layer.backward(upstream)
    d_W_analytic = layer.d_W
    d_b_analytic = layer.d_b

    d_W_numeric = numerical_gradient(layer.W, compute_loss) / batch
    d_b_numeric = numerical_gradient(layer.b, compute_loss) / batch
    d_x_numeric = numerical_gradient(x, compute_loss)

    print("Dense d_W relative error:", relative_error(d_W_analytic, d_W_numeric))
    print("Dense d_b relative error:", relative_error(d_b_analytic, d_b_numeric))
    print("Dense d_x relative error:", relative_error(d_x_analytic, d_x_numeric))


if __name__ == "__main__":
    check_conv2d()
    check_dense()
