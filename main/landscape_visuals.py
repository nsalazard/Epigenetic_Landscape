import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import subprocess
import shutil
import os

from cmcrameri import cm as scm
from matplotlib.colors import ListedColormap, BoundaryNorm, CenteredNorm
# from skimage.measure import label

plt.rcParams.update({'figure.dpi': 100})  # Change to 200 for high res figures

# to use for modules if colored by type
fp_type_colors = {
    'Node': 'tab:green',
    'UnstableNode': 'tab:blue',
    'Center': 'tab:purple',
    'NegCenter': 'hotpink',
}

# to use for modules if colored by order in the module_list
order_colors = (
    '#000000', #black
    '#DC143C', #Crimson
    '#00008B', #Dark Blue
    '#228B22', #Forest Green
    'indianred',
    'tab:blue',
    'tab:green',
    'gold',
    'tab:purple',
    'm',
)

# ________________________________________________________________________________________________________________

# TODO: main visualizing function with 4 panels
def visualize_all(landscape, xx, yy, times, density=0.5, color_scheme='fp_types',
                  plot_velocities=True, plot_nullclines=True,
                  plot_traj=True, traj_times=(0., 100., 150), plot_start=50, traj_init_cond=(0., 1.), traj_noise=0., ):

    """
    Plot 4 panels: potential contour plot, rotational potential contour plot, flow plot with module circles,
    and flow plot with velocity magnitude
    :param landscape:
    :param xx:
    :param yy:
    :param times:
    :param density:
    :param color_scheme:
    :param plot_velocities:
    :param plot_nullclines:
    :param plot_traj:
    :param traj_times:
    :param plot_start:
    :param traj_init_cond:
    :param traj_noise:
    :return:
    """
    dX, dY = np.zeros((len(times), *xx.shape)), np.zeros((len(times), *xx.shape))

    for it in range(len(times)):

        (dX[it], dY[it]), potential, rot_potential = landscape(times[it], (xx, yy), return_potentials=True)

        circles = []
        for i, module in enumerate(landscape.module_list):
            V, sig, A = module.get_current_pars(times[it], landscape.regime, *landscape.morphogen_times)
            if color_scheme == 'fp_types':
                color = fp_type_colors[module.__class__.__name__]
            elif color_scheme == 'order':
                color = order_colors[i]
            else:
                color = 'grey'
            circles.append(plt.Circle((module.x, module.y), 1.18 * sig, color=color,
                                      fill=True, alpha=0.3 * np.sqrt(A), clip_on=True, linewidth=0))

        fig, ax = plt.subplots(1, 4, figsize=(18, 4))
        ax[0].imshow(potential, cmap=scm.cork.reversed(), origin='lower', norm=CenteredNorm(0),
                     extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
        ax[0].contour(xx, yy, -potential, origin='lower', colors='w')

        vrange = (np.max(rot_potential) - np.min(rot_potential))/2.
        if vrange == 0.:
            vrange = 1.
        ax[1].imshow(rot_potential, cmap='RdBu_r', origin='lower', norm=CenteredNorm(0, vrange),
                     extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))
        ax[1].contour(xx, yy, rot_potential, colors='w', linestyles='solid', origin='lower')

        for iax in range(4):
            ax[iax].set_xticks([])
            ax[iax].set_yticks([])
            ax[iax].set_xlim((np.min(xx), np.max(xx)))
            ax[iax].set_ylim((np.min(yy), np.max(yy)))

            # for i, module in enumerate(landscape.module_list):
            #     ax[iax].scatter(module.x, module.y, marker='x', c='k')
        circles_ax = ax[2]
        stream_ax = ax[3]

        for i in range(len(landscape.module_list)):
            circles_ax.add_patch(copy(circles[i]))

        if plot_velocities:
            velocities_sq = dX[it] ** 2 + dY[it] ** 2
            velocities = np.sqrt(velocities_sq)
            # print('Min velocity:', round(np.min(velocities), 3), ', Max:', round(np.max(velocities), 3),
            #       ', Mean:', round(np.mean(velocities), 3), ', Median:', round(np.median(velocities), 3))

            stream_ax.imshow(velocities, alpha=0.5, cmap='Greys', origin='lower',
                             extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)))

            # An attempt to plot fixed points - often ends up missing some points
            # fp_labels, nlabels = label(velocities_sq < 1e-3, return_num=True)
            # for l in range(nlabels):
            #     # if np.sum(fp_labels == l) <= 50:
            #     fp = velocities_sq == np.min(velocities_sq[fp_labels == l])
            #     # if np.sum(fp_labels == l) > 20:
            #     #     fp = (velocities_sq < 5e-4) * fp_labels == l
            #     # else:
            #     #     fp = fp_labels == l
            #     stream_ax.scatter(xx[fp], yy[fp], marker='o', s=50, color='gold', edgecolor=None, zorder=10)

        circles_ax.streamplot(xx, yy, dX[it], dY[it], density=density, arrowsize=2., arrowstyle='->',
                              linewidth=1,
                              color='grey')
        stream_ax.streamplot(xx, yy, dX[it], dY[it], density=density, arrowsize=2., arrowstyle='->',
                             linewidth=1,
                             color='grey')

        if plot_nullclines:
            circles_ax.contour(xx, yy, dX[it], (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
            circles_ax.contour(xx, yy, dY[it], (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)
            stream_ax.contour(xx, yy, dX[it], (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
            stream_ax.contour(xx, yy, dY[it], (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)

        if plot_traj:
            # calculate a trajectory in frozen landscape
            #landscape.init_cells(1, traj_init_cond, noise=traj_noise)
            landscape.run_cells(noise=traj_noise, ndt=50, frozen=True, t_freeze=times[it])
            stream_ax.plot(landscape.cell.Positions[0, 0, plot_start:], landscape.cell.Positions[1, 0, plot_start:], lw=2.5, color='forestgreen')

        plt.show()

    return dX, dY


def plot_cells(landscape, L, colors=None):
    """ Plot the current cell locations and states """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    coord = landscape.cell.pos
    states = landscape.cell.States
    if colors is None:
        colors = order_colors
    cmap_state = ListedColormap(colors)
    norm_state = BoundaryNorm(np.arange(len(colors)+1) - 0.5, cmap_state.N)
    ax.scatter(landscape.cell.Positions[0,:,-1], landscape.cell.Positions[1,:,-1], s=8, alpha=0.3, c=landscape.cell.States[:,-1], cmap=cmap_state, norm=norm_state, edgecolors=None)
    ax.set_xlim([-L, L])
    ax.set_ylim([-L, L])
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_aspect(1)
    plt.show()


def get_and_plot_traj(landscape, L, noise=0.5, ndt=50, frozen=False, t_freeze=None, colors=None):
    """ Integrate trajectories for cells and visualize them in 2 panels:
    colored by timepoint and colored be cell state """
    if colors is None:
        colors = order_colors
    cmap_state = ListedColormap(colors)
    norm_state = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, cmap_state.N)
    cmap_time = 'viridis'
    landscape.run_cells(noise=noise, ndt=ndt, frozen=frozen, t_freeze=t_freeze)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].scatter(landscape.cell.Positions[0, :, :], landscape.cell.Positions[1, :, :], s=6, alpha=0.2, c=np.tile(np.arange(landscape.cell.nt), (landscape.cell.States.shape[0], 1)), cmap=cmap_time, edgecolor=None)
    ax[1].scatter(landscape.cell.Positions[0, :, :], landscape.cell.Positions[1, :, :], s=6, alpha=0.2, c=landscape.cell.States, cmap=cmap_state, norm=norm_state, edgecolors=None)

    ax[0].set_xlim([-L, L])
    ax[0].set_ylim([-L, L])
    ax[1].set_xlim([-L, L])
    ax[1].set_ylim([-L, L])
    #ax[0].set_xticks([])
    #ax[0].set_yticks([])
    #ax[1].set_xticks([])
    #ax[1].set_yticks([])
    plt.show()


# MARK: - visualize_landscape

def visualize_landscape(landscape, xx, yy, regime, color_scheme='fp_types',size_x=5, size_y=5):
    """ Simple visualization of landscape flow and modules in one regime. """
    density = 0.5
    curl = np.zeros((len(landscape.module_list)), dtype='bool')
    circles = []
    for i, module in enumerate(landscape.module_list):
        if module.__class__.__name__ == 'Center' or module.__class__.__name__ == 'NegCenter':
            curl[i] = 1

    for i, module in enumerate(landscape.module_list):
        if module.a.size == 1 and module.s.size == 1 and regime == 0:
            sig = module.s
            A = module.a
        else:
            sig = module.s[regime]
            A = module.a[regime]
        if color_scheme == 'fp_types':
            color = fp_type_colors[module.__class__.__name__]
        elif color_scheme == 'order':
            color = order_colors[i]
        else:
            color = 'grey'

        circles.append(plt.Circle((module.x, module.y), 1.18 * sig, color=color,
                                  fill=True, alpha=0.3 * np.sqrt(A), clip_on=True, linewidth=0))
    morphogen_times = landscape.morphogen_times
    landscape.morphogen_times = np.arange(landscape.n_regimes) + 0.5
    (dX, dY), potential, rot_potential = landscape(float(regime), (xx, yy), return_potentials=True)

    fig, stream_ax = plt.subplots(1, 1, figsize=(size_x, size_y))
    circles_ax = stream_ax

    for i in range(len(landscape.module_list)):
        circles_ax.add_patch(copy(circles[i]))
        circles_ax.set_xlim((np.min(xx), np.max(xx)))
        circles_ax.set_ylim((np.min(yy), np.max(yy)))

    stream_ax.streamplot(xx, yy, dX, dY, density=density, arrowsize=2., arrowstyle='->', linewidth=1,
                         color='grey')
    stream_ax.contour(xx, yy, dX, (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
    stream_ax.contour(xx, yy, dY, (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)

    stream_ax.set_xlim([np.min(xx), np.max(xx)])
    stream_ax.set_ylim([np.min(yy), np.max(yy)])
    stream_ax.set_xticks([])
    stream_ax.set_yticks([])
    landscape.morphogen_times = morphogen_times
    plt.show()
    return fig

# MARK: - visualize_landscape_t

def visualize_landscape_t(landscape, xx, yy, t, color_scheme='fp_types', traj_times=None, traj_init_cond=(0., 0.), traj_start=0, size_x=4, size_y=4, ndt= 100):
    """ Visualize the flow and modules at time t, with optional integrated trajectory in the frozen landscape. """
    density = 0.5
    curl = np.zeros((len(landscape.module_list)), dtype='bool')
    circles = []
    for i, module in enumerate(landscape.module_list):
        if module.__class__.__name__ == 'Center' or module.__class__.__name__ == 'NegCenter':
            curl[i] = 1

    for i, module in enumerate(landscape.module_list):
        V, sig, A = module.get_current_pars(t, landscape.regime, *landscape.morphogen_times)
        if color_scheme == 'fp_types':
            color = fp_type_colors[module.__class__.__name__]
        elif color_scheme == 'order':
            color = order_colors[i]
        else:
            color = 'grey'

        circles.append(plt.Circle((module.x, module.y), 1.18 * sig, color=color,
                                  fill=True, alpha=0.3 * np.sqrt(A), clip_on=True, linewidth=0))
    (dX, dY), potential, rot_potential = landscape(t, (xx, yy), return_potentials=True)

    fig, stream_ax = plt.subplots(1, 1, figsize=(size_x, size_y))
    circles_ax = stream_ax

    for i in range(len(landscape.module_list)):
        circles_ax.add_patch(copy(circles[i]))
        circles_ax.set_xlim((np.min(xx), np.max(xx)))
        circles_ax.set_ylim((np.min(yy), np.max(yy)))

    stream_ax.streamplot(xx, yy, dX, dY, density=density, arrowsize=2., arrowstyle='->', linewidth=1,color='grey')
    stream_ax.contour(xx, yy, dX, (0,), colors=('k',), linestyles='-', linewidths=1.5, alpha=0.7)
    stream_ax.contour(xx, yy, dY, (0,), colors=('k',), linestyles='--', linewidths=1.5, alpha=0.7)

    if traj_times is not None:
        landscape.run_cells(noise=0., ndt=ndt, frozen=True,t_freeze=t)
        stream_ax.plot(landscape.cell.Positions[0, 0, traj_start:], landscape.cell.Positions[1, 0, traj_start:], lw=2.5, color='forestgreen')

    stream_ax.set_xlim([np.min(xx), np.max(xx)])
    stream_ax.set_ylim([np.min(yy), np.max(yy)])
    stream_ax.set_xticks([])
    stream_ax.set_yticks([])
    # landscape.morphogen_times = morphogen_times
    # plt.show()
    return fig, stream_ax

# MARK: - video_landscape
# TODO: Fix problems regarding google colab

def video_landscape(landscape, xx, yy, traj_times=(0,0,0), color_scheme='fp_types', plot_start=0, size_x=4, size_y=4,ndt= 100, noise_init=2., noise_run=0.2, tstep = 0, colors=None, video_name = 'Landscape_video', same_time= True,  measure='dist', dwl=False):

    if traj_times is not None:
        t0,tf = traj_times[0],traj_times[1]
    else:
        t0,tf = landscape.cell.t0, landscape.cell.tf
    frames = range(landscape.cell.nt)  # Frames for the animation
    # Directory to save frames
    output_dir = '/content/IMGs/'
    os.makedirs(output_dir, exist_ok=True)
    if colors is None:
        colors = order_colors
    cmap_state = ListedColormap(colors)
    norm_state = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, cmap_state.N)

    landscape.run_cells(noise=noise_run, ndt=ndt, same_time= same_time, measure=measure)
    Delta_t = (tf - t0) / landscape.cell.nt
    t= t0

    for frame in frames:
        t += Delta_t
        #print(t)
        fig, ax = visualize_landscape_t(landscape, xx, yy, t=int(t), traj_times=None,  color_scheme='fp_types',size_x=10, size_y=5)

        ax.scatter(landscape.cell.Positions[0, :, frame], landscape.cell.Positions[1, :, frame], s=13, alpha=1, c=landscape.cell.States[:,frame], cmap=cmap_state, norm=norm_state, edgecolors=None)

        fig.savefig(f'{output_dir}/frame_{frame:03d}.png')
        plt.close(fig)
        #print(frame)

    cmd = f"ffmpeg -r 10 -pattern_type glob -i '{output_dir}/frame_*.png' -c:v libx264 -pix_fmt yuv420p /content/{video_name}.mp4"
    subprocess.run(cmd, shell=True)

    shutil.rmtree(output_dir)

   # if dwl:
   #     files.download(f'/content/{video_name}.mp4')

# MARK: - visualize_potential

def visualize_potential(landscape, xx, yy, regime, color_scheme='fp_types', elev=None, azim=None, offset=2,
                        cmap_center=None, rot=False, scatter=False, zlim=None, tilt_par = None):
    curl = np.zeros((len(landscape.module_list)), dtype='bool')
    # circles = []
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(6, 6))
    ax.view_init(elev=elev, azim=azim)
    morphogen_times = landscape.morphogen_times
    landscape.morphogen_times = np.arange(landscape.n_regimes) + 0.5
    (dX, dY), potential, rot_potential = landscape(float(regime), (xx, yy), return_potentials=True)
    print(f'Tilt:{landscape.tilt_var} in t: {float(regime)}')
    if cmap_center is None:
        cmap_center = potential[0, 0]
    if rot:
        potential = rot_potential
        cmap = 'RdBu'
    else:
        cmap = scm.cork.reversed()

    if zlim is None:
        ax.set_zlim([np.min(potential) - offset, np.max(potential) + 2])
        zlow = np.min(potential) - offset
    else:
        ax.set_zlim(zlim)
        zlow = zlim[0]
    ax.contour(xx, yy, potential, zdir='z', offset=zlow, cmap=cmap, norm=CenteredNorm(cmap_center))
    ax.plot_surface(xx, yy, potential, cmap=cmap, linewidth=0, antialiased=False, norm=CenteredNorm(cmap_center))
    # if wind:
    #     right = rot_potential.copy()
    #     left = rot_potential.copy()
    #     right[rot_potential < 0] = 0
    #     left[rot_potential > 0] = 0
    #     ax.contour(xx, yy, right, zdir='z', offset=np.max(potential), cmap='RdBu', norm=CenteredNorm(0), zorder=10)
    #     ax.contour(xx, yy, np.abs(left), zdir='z', offset=np.max(potential), cmap='RdBu_r', norm=CenteredNorm(0),
    #                zorder=10)

    if scatter:
        for i, module in enumerate(landscape.module_list):
            if module.__class__.__name__ == 'Center' or module.__class__.__name__ == 'NegCenter':
                curl[i] = 1
            if color_scheme == 'fp_types':
                color = fp_type_colors[module.__class__.__name__]
            elif color_scheme == 'order':
                color = order_colors[i]
            else:
                color = 'grey'
            ax.scatter(module.x, module.y, zlow, s=25, color=color, marker='D', zorder=20)

    landscape.morphogen_times = morphogen_times

    ax.set_xticks([])
    ax.set_yticks([])
    ax.zaxis.set_tick_params(color='white')
    ax.set_zticklabels([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # plt.tight_layout()
    # plt.show()
    return fig


