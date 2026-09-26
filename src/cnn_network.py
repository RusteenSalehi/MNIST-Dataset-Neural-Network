import numpy as np


def naive_convolve2d(image, kernel):
    """Single-channel, single-kernel, no-padding, stride-1 convolution.

    For understanding only -- not used by the layer classes below, which use
    im2col instead.
    """
    # image: (H, W)
    # kernel: (kH, kW)
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    out_height = height - kernel_height + 1
    out_width = width - kernel_width + 1

    # output: (out_height, out_width)
    output = np.zeros((out_height, out_width))
    for i in range(out_height):
        for j in range(out_width):
            patch = image[i:i + kernel_height, j:j + kernel_width]  # (kH, kW)
            output[i, j] = np.sum(patch * kernel)
    return output


def im2col(x, kernel_height, kernel_width, stride=1):
    """Rearrange sliding patches of x into columns of a single matrix.

    x: (batch, channels, height, width)
    returns: col (channels * kernel_height * kernel_width, batch * out_h * out_w),
             out_h, out_w
    """
    batch, channels, height, width = x.shape
    out_h = (height - kernel_height) // stride + 1
    out_w = (width - kernel_width) // stride + 1

    col = np.zeros((channels * kernel_height * kernel_width, batch * out_h * out_w))
    col_index = 0
    for b in range(batch):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                # patch: (channels, kernel_height, kernel_width)
                patch = x[b, :, h_start:h_start + kernel_height, w_start:w_start + kernel_width]
                col[:, col_index] = patch.reshape(-1)
                col_index += 1
    return col, out_h, out_w


def col2im(col, x_shape, kernel_height, kernel_width, stride=1):
    """Inverse of im2col: scatter-add columns back into overlapping patches.

    col: (channels * kernel_height * kernel_width, batch * out_h * out_w)
    x_shape: (batch, channels, height, width), shape of the original input
    returns: d_x, shape x_shape
    """
    batch, channels, height, width = x_shape
    out_h = (height - kernel_height) // stride + 1
    out_w = (width - kernel_width) // stride + 1

    d_x = np.zeros(x_shape)
    col_index = 0
    for b in range(batch):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                # patch_grad: (channels, kernel_height, kernel_width)
                patch_grad = col[:, col_index].reshape(channels, kernel_height, kernel_width)
                d_x[b, :, h_start:h_start + kernel_height, w_start:w_start + kernel_width] += patch_grad
                col_index += 1
    return d_x


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        fan_in = in_channels * kernel_size * kernel_size
        # W: (out_channels, in_channels, kernel_size, kernel_size)
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2 / fan_in)
        self.b = np.random.randn(out_channels, 1)

        # Cached for backward
        self.x = None
        self.col = None
        self.out_h = None
        self.out_w = None

    def forward(self, x):
        # x: (batch, in_channels, height, width)
        batch = x.shape[0]
        self.x = x

        # col: (in_channels * kernel_size * kernel_size, batch * out_h * out_w)
        self.col, self.out_h, self.out_w = im2col(x, self.kernel_size, self.kernel_size, self.stride)

        # w_flat: (out_channels, in_channels * kernel_size * kernel_size)
        w_flat = self.W.reshape(self.out_channels, -1)

        # out: (out_channels, batch * out_h * out_w)
        out = np.matmul(w_flat, self.col) + self.b

        # (out_channels, batch, out_h, out_w) -> (batch, out_channels, out_h, out_w)
        out = out.reshape(self.out_channels, batch, self.out_h, self.out_w)
        out = out.transpose(1, 0, 2, 3)
        return out

    def backward(self, d_out):
        # d_out: (batch, out_channels, out_h, out_w)
        batch = self.x.shape[0]

        # Undo the forward reshape/transpose: back to (out_channels, batch * out_h * out_w)
        d_out_flat = d_out.transpose(1, 0, 2, 3).reshape(self.out_channels, -1)

        w_flat = self.W.reshape(self.out_channels, -1)  # (out_channels, in_channels*kH*kW)

        # d_w_flat: (out_channels, in_channels*kH*kW), summed over batch and spatial positions
        d_w_flat = np.matmul(d_out_flat, self.col.T)
        self.d_W = d_w_flat.reshape(self.W.shape) / batch
        self.d_b = np.sum(d_out_flat, axis=1, keepdims=True) / batch

        # d_col: (in_channels*kH*kW, batch*out_h*out_w), undivided, flows further back
        d_col = np.matmul(w_flat.T, d_out_flat)

        # Scatter back into overlapping input patches
        d_x = col2im(d_col, self.x.shape, self.kernel_size, self.kernel_size, self.stride)
        return d_x


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, d_out):
        return d_out * self.mask


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

        # Cached for backward
        self.x_shape = None
        self.max_mask = None
        self.out_h = None
        self.out_w = None

    def forward(self, x):
        # x: (batch, channels, height, width)
        batch, channels, height, width = x.shape
        pool = self.pool_size
        stride = self.stride
        out_h = (height - pool) // stride + 1
        out_w = (width - pool) // stride + 1

        self.x_shape = x.shape
        self.out_h = out_h
        self.out_w = out_w
        self.max_mask = np.zeros_like(x, dtype=bool)

        # output: (batch, channels, out_h, out_w)
        output = np.zeros((batch, channels, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                # window: (batch, channels, pool, pool)
                window = x[:, :, h_start:h_start + pool, w_start:w_start + pool]
                window_flat = window.reshape(batch, channels, -1)  # (batch, channels, pool*pool)

                output[:, :, i, j] = np.max(window_flat, axis=2)

                # Record which position in the window held the max, for backward
                max_index = np.argmax(window_flat, axis=2)  # (batch, channels)
                window_mask = np.zeros((batch, channels, pool * pool), dtype=bool)
                b_idx, c_idx = np.meshgrid(np.arange(batch), np.arange(channels), indexing="ij")
                window_mask[b_idx, c_idx, max_index] = True
                self.max_mask[:, :, h_start:h_start + pool, w_start:w_start + pool] = \
                    window_mask.reshape(batch, channels, pool, pool)

        return output

    def backward(self, d_out):
        # d_out: (batch, channels, out_h, out_w)
        pool = self.pool_size
        stride = self.stride

        d_x = np.zeros(self.x_shape)
        for i in range(self.out_h):
            for j in range(self.out_w):
                h_start = i * stride
                w_start = j * stride
                # window_mask: (batch, channels, pool, pool), True at the winning position
                window_mask = self.max_mask[:, :, h_start:h_start + pool, w_start:w_start + pool]
                # broadcast d_out[:, :, i, j] (batch, channels) over the pool x pool window
                d_x[:, :, h_start:h_start + pool, w_start:w_start + pool] += (
                    window_mask * d_out[:, :, i, j][:, :, np.newaxis, np.newaxis]
                )
        return d_x


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        # x: (batch, channels, height, width) -> (channels * height * width, batch)
        self.input_shape = x.shape
        batch = x.shape[0]
        flat = x.reshape(batch, -1)  # (batch, features)
        return flat.T  # (features, batch)

    def backward(self, d_out):
        # d_out: (features, batch) -> (batch, features) -> input_shape
        return d_out.T.reshape(self.input_shape)


class Dense:
    def __init__(self, in_features, out_features):
        # W: (out_features, in_features)
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2 / in_features)
        self.b = np.random.randn(out_features, 1)

        self.x = None

    def forward(self, x):
        # x: (in_features, batch)
        self.x = x
        return np.matmul(self.W, x) + self.b  # (out_features, batch)

    def backward(self, d_out):
        # d_out: (out_features, batch)
        batch = self.x.shape[1]
        self.d_W = np.matmul(d_out, self.x.T) / batch
        self.d_b = np.sum(d_out, axis=1, keepdims=True) / batch
        d_x = np.matmul(self.W.T, d_out)  # (in_features, batch), undivided
        return d_x


class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None

    def forward(self, z):
        # z: (num_classes, batch)
        z_shifted = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shifted)
        self.probs = exp_z / np.sum(exp_z, axis=0, keepdims=True)
        return self.probs

    @staticmethod
    def loss(probs, y):
        # probs, y: (num_classes, batch)
        epsilon = 1e-10
        batch_size = y.shape[1]
        return -np.sum(y * np.log(probs + epsilon)) / batch_size

    def backward(self, y):
        # y: (num_classes, batch), one-hot targets
        return self.probs - y  # (num_classes, batch), undivided by batch size


class Network:
    """Conv2D(1->8) -> ReLU -> Pool -> Conv2D(8->16) -> ReLU -> Pool
    -> Flatten -> Dense(->64) -> ReLU -> Dense(->10) -> Softmax."""

    def __init__(self):
        self.conv1 = Conv2D(1, 8, 3)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2, 2)

        self.conv2 = Conv2D(8, 16, 3)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2, 2)

        self.flatten = Flatten()
        self.dense1 = Dense(16 * 5 * 5, 64)
        self.relu3 = ReLU()
        self.dense2 = Dense(64, 10)
        self.softmax = SoftmaxCrossEntropy()

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        out = self.conv1.forward(x)      # (batch, 8, 26, 26)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)    # (batch, 8, 13, 13)

        out = self.conv2.forward(out)    # (batch, 16, 11, 11)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)    # (batch, 16, 5, 5)

        out = self.flatten.forward(out)  # (400, batch)
        out = self.dense1.forward(out)   # (64, batch)
        out = self.relu3.forward(out)
        out = self.dense2.forward(out)   # (10, batch)

        return self.softmax.forward(out)  # (10, batch)

    def loss(self, probs, y):
        return self.softmax.loss(probs, y)

    def backward(self, y, lr):
        d = self.softmax.backward(y)
        d = self.dense2.backward(d)
        d = self.relu3.backward(d)
        d = self.dense1.backward(d)
        d = self.flatten.backward(d)
        d = self.pool2.backward(d)
        d = self.relu2.backward(d)
        d = self.conv2.backward(d)
        d = self.pool1.backward(d)
        d = self.relu1.backward(d)
        self.conv1.backward(d)

        for layer in (self.conv1, self.conv2, self.dense1, self.dense2):
            layer.W -= lr * layer.d_W
            layer.b -= lr * layer.d_b


if __name__ == "__main__":
    from data_loader import load_mnist

    x_train, _, _, _ = load_mnist()
    image = x_train[0, 0]  # (28, 28), first training image, single channel

    vertical_edge_kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ], dtype=np.float32)

    output = naive_convolve2d(image, vertical_edge_kernel)

    print(f"Input shape: {image.shape}")
    print(f"Kernel shape: {vertical_edge_kernel.shape}")
    print(f"Output shape: {output.shape}")
