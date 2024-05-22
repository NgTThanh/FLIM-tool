import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from .core import sofism


def spad_to_img(data):
    """
    Convert list of SPAD array detector counts to an image
    """
    res = np.zeros((10, 10))
    for i in range(0, 5, 2):
        for j in range(5):
            res[2 * i:2 * (i + 1), 2 * j:2 * (j + 1)] = data[5 * i + j - i // 2]
    for i in range(1, 5, 2):
        for j in range(4):
            res[2 * i:2 * (i + 1), 2 * j + 1:2 * (j + 1) + 1] = data[5 * i + j]
    return res[:, 1:9]


def plot_z_profile(img, params):
    if params is None:
        return 0
    else:
        fig = plt.figure(figsize=(12, 5))
        fig.add_subplot(111)
        plt.plot(img.sum(axis=0))
        plt.grid()
        plt.xlabel("Z [$\mu$m]", fontsize=18)
        plt.ylabel("Count sum")
        # initial position
        zo = float(params.loc['zo'].values[0])
        # resolution
        dz = float(params.loc['dz'].values[0])
        # n steps
        nz = float(params.loc['nz'].values[0])
        labels1 = np.round(zo + np.arange(-nz // 2, nz // 2, 1) * dz, 2)
        plt.gca().set_xticks(np.arange(0, nz, 1))
        plt.gca().set_xticklabels(labels1, fontsize=14)
        plt.tight_layout()
        plt.show()


def spad_sum_fig(img, params=None):
    """
    Creates matplotlib heatmap (ax.imshow) of data in img (2d np.array)
    and converts to QImage; mpl drawing outside of main thread fails
    params:
        measurement parameters, if not None, used for labeling axes
    """
    time.sleep(0.02)
    fig = plt.figure(figsize=(8, 6))
    # canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    dev = params.loc["dev"].values[0]
    im_roi = img
    # tiles
    ix = int(params.loc["ix"].values[0])
    iy = int(params.loc["iy"].values[0])
    if ix == 1 and iy == 1:
        xo = float(params.loc['xo'].values[0])
        yo = float(params.loc['yo'].values[0])
    else:
        xo = float(params.loc['tileX0'].values[0])
        yo = float(params.loc['tileY0'].values[0])
    zo = float(params.loc['zo'].values[0])

    # resolution
    dx = float(params.loc['dx'].values[0])
    dy = float(params.loc['dy'].values[0])
    dz = float(params.loc['dz'].values[0])
    # dev = str(params.loc['dev'].values[0])

    nx = float(params.loc['nx'].values[0])
    ny = float(params.loc['ny'].values[0])
    nz = float(params.loc['nz'].values[0])

    if nx > 1 and ny > 1:
        if dev == 'SPADcontinuous':
            p1, p2 = ix * nx, iy * ny
            ax.set_ylabel("y [$\mu$m]", fontsize=18)
            ax.set_xlabel("x [$\mu$m]", fontsize=18)
            labels1 = np.round(xo + np.arange(0, p1, (p1 // 5)) * dx, 2)
            labels2 = np.round(yo + np.arange(0, p2, p2 // 5) * dy, 2)
            aspect = dy / dx
        else:
            p1, p2 = ix * nx, iy * ny
            ax.set_ylabel("y [$\mu$m]", fontsize=18)
            ax.set_xlabel("x [$\mu$m]", fontsize=18)
            labels1 = np.round(yo + np.arange(p1, 0, -(p1 // 5)) * dx, 2)
            labels2 = np.round(xo + np.arange(0, p2, p2 // 5) * dy, 2)
            aspect = dy / dx
    elif nx == 1:
        print("YZ image")
        p1, p2 = nz, ny
        p1o, p2o = zo, yo
        ax.set_ylabel("Y [$\mu$m]", fontsize=18)
        labels2 = np.round(p2o + np.arange(0, p2, p2 // 5) * dy, 2)
        ax.set_xlabel("Z [$\mu$m]", fontsize=18)
        labels1 = np.round(p1o + np.arange(-p1 // 2, p1 // 2, p1 // 5) * dz, 2)
        aspect = dy / dz
    else:
        print("XZ image")
        p2, p1 = nx, nz
        labels2 = np.round(xo + np.arange(p2, 0, -(p2 // 5)) * dx, 2)
        ax.set_ylabel("X [$\mu$m]", fontsize=18)
        labels1 = np.round(zo + np.arange(-p1 // 2, p1 // 2, p1 // 5) * dz, 2)
        ax.set_xlabel("Z [$\mu$m]", fontsize=18)
        aspect = dx / dz

    # axes ticks and labels
    ax.set_yticks(np.arange(0, p2, p2 // 5))
    ax.set_yticklabels(labels2, fontsize=14)
    ax.set_xticks(np.arange(0, p1, p1 // 5))
    ax.set_xticklabels(labels1, fontsize=14)

    imm = ax.imshow(im_roi, cmap='inferno', aspect=aspect)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.15)
    fig.colorbar(imm, cax=cax, orientation='vertical')
    fig.tight_layout()

    return fig


def visualize(data, meas_path, params):
    fig = plt.figure(figsize=(12, 10))
    for i in range(5):
        for j in range(5):
            if 5 * i + j == 23:
                fig.add_subplot(5, 5, 24)
                plt.imshow(data.sum(axis=(0, 3)), cmap='inferno')
                plt.colorbar()
            elif 5 * i + j == 24:
                fig.add_subplot(5, 5, 25)
                plt.imshow(spad_to_img(data.sum(axis=(1, 2, 3))), cmap='inferno')
                plt.colorbar()
            else:
                fig.add_subplot(5, 5, 5 * i + j + 1)
                plt.imshow(data[5 * i + j, :, :, :].sum(axis=2), cmap='inferno')
                plt.colorbar()
    plt.tight_layout()
    plt.savefig(meas_path.format("det_images.png"))
    plt.close()  # Monika
    fig = spad_sum_fig(data.sum(axis=(0, 3)), params)
    fig.savefig(meas_path.format("sum.png"))
    plt.close()  # Monika
    print("Images generated and saved in: ", meas_path)

    plt.plot(data[:, :, :, :].sum(axis=(0, 1, 2)))
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("Sum of counts")
    plt.tight_layout()
    # plt.show()
    fig.savefig(meas_path.format("avg_time_trace.png"))

    return fig, data.sum(axis=(0, 3))


def test_shifts(data, shifts):
    ops = ["X $\Rightarrow$ X, Y $\Rightarrow$ Y", "X $\Rightarrow$ -X, Y $\Rightarrow$ Y", "X $\Rightarrow$ X, Y $\Rightarrow$ -Y", "X $\Rightarrow$ -X, Y $\Rightarrow$ -Y",
           "X $\Rightarrow$ Y, Y $\Rightarrow$ X", "X $\Rightarrow$ -Y, Y $\Rightarrow$ X", "X $\Rightarrow$ Y, Y $\Rightarrow$ -X", "X $\Rightarrow$ -Y, Y $\Rightarrow$ -X"]
    ism = [None] * 8
    for i in range(len(ops)):
        curr_shifts = np.array(shifts)
        if i == 0:
            pass
        elif i == 1:
            curr_shifts[:, 0] = -shifts[:, 0]
        elif i == 2:
            curr_shifts[:, 1] = -shifts[:, 1]
        elif i == 3:
            curr_shifts[:, 0] = -shifts[:, 0]
            curr_shifts[:, 1] = -shifts[:, 1]
        elif i == 4:
            curr_shifts[:, 0] = shifts[:, 1]
            curr_shifts[:, 1] = shifts[:, 0]
        elif i == 5:
            curr_shifts[:, 0] = -shifts[:, 1]
            curr_shifts[:, 1] = shifts[:, 0]
        elif i == 6:
            curr_shifts[:, 0] = shifts[:, 1]
            curr_shifts[:, 1] = -shifts[:, 0]
        elif i == 7:
            curr_shifts[:, 0] = -shifts[:, 1]
            curr_shifts[:, 1] = -shifts[:, 0]
        res = sofism(data, None, curr_shifts)
        clsm, ism[i] = res.values()
        
    fig = plt.figure(figsize=(18, 15))
    fig.add_subplot(331)
    plt.imshow(clsm, cmap='inferno')
    plt.gca().set_title("CLSM")
    plt.colorbar()
    for i in range(len(ism)):
        fig.add_subplot(332+i)
        plt.imshow(ism[i], cmap='inferno')
        plt.gca().set_title(f"{i}: {ops[i]}")
        plt.colorbar()
        plt.tight_layout()
    plt.show()
    return fig



