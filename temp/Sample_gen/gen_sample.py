import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkAgg')
import temp.Sample_gen.mode_default as gm


gen_mode = gm.gen_mode







def gen_sample(N_track):

    # Generate the hit data by random sampling phi theta
    rand_angle = np.random.rand(N_track, 2)
    phi = rand_angle[:, 0] * (gen_mode["phi_range"][1] - gen_mode["phi_range"][0]) + gen_mode["phi_range"][0]
    cos_theta = rand_angle[:, 1] * (gen_mode["ctheta_range"][1] - gen_mode["ctheta_range"][0]) + \
                gen_mode["ctheta_range"][0]
    sin_theta = np.sqrt(1 - cos_theta ** 2)

    r = np.zeros((N_track, 4))
    x = np.zeros((N_track, 4))
    y = np.zeros((N_track, 4))
    for idx, z in enumerate(gen_mode["z_range"]):
        r[:, idx] = z / cos_theta
        x[:, idx] = r[:, idx] * np.cos(phi) * sin_theta + np.random.normal(0, gen_mode['x_bias'], N_track)
        y[:, idx] = r[:, idx] * np.sin(phi) * sin_theta + np.random.normal(0, gen_mode['y_bias'], N_track)

    particle_index = np.linspace(1, N_track, N_track, dtype=int)

    hit = np.zeros((N_track * 4, 5))
    hit[:, 0] = np.linspace(0, 4 * N_track-1, 4 * N_track, dtype=int)
    for layer, z in enumerate(gen_mode["z_range"]):
        hit[layer * N_track:(layer + 1) * N_track, 1] = x[:, layer]
        hit[layer * N_track:(layer + 1) * N_track, 2] = y[:, layer]
        hit[layer * N_track:(layer + 1) * N_track, 3] = z
        hit[layer * N_track:(layer + 1) * N_track, 4] = particle_index
    hit_df = pd.DataFrame(hit, columns=['hit_id', 'x', 'y', 'z', 'particle_index'])
    hit_df['particle_index'] = hit_df['particle_index'].astype(int)
    hit_df['hit_id'] = hit_df['hit_id'].astype(int)


    N_noise = int(N_track * gen_mode['noise_ratio'])
    if N_noise > 0 and N_noise is not None:
        hit_df = gen_noise(hit_df, N_track, N_noise)

    return hit_df


def gen_noise(hit_df, N_track, N_noise):
    N_noise = N_noise * 4
    if N_noise == 0:
        return hit_df
    squrerange = np.max(gen_mode["z_range"]) / np.tan(gen_mode["ctheta_range"][0]) * 1.2
    x = np.random.random(N_noise) * squrerange * 2 - squrerange
    y = np.random.random(N_noise) * squrerange * 2 - squrerange
    z = np.random.randint(0, 4, N_noise)
    # print(z)
    z = np.array(gen_mode["z_range"])[z]
    # print(z)
    particle_index = 0
    df = pd.DataFrame({'hit_id': np.linspace(N_track * 4, N_noise + N_track * 4-1, N_noise, dtype=int),
                       'x': x,
                       'y': y,
                       'z': z,
                       'particle_index': particle_index})
    hit_df = pd.concat([hit_df, df], ignore_index=True)
    # print(hit_df)
    return hit_df




def visualzie_sample(sample):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sample['x'], sample['y'], sample['z'])
    ax.scatter(0, 0, 0)

    # 创建 x 和 y 数据
    squrerange = np.max(gen_mode["z_range"]) / np.tan(gen_mode["ctheta_range"][0]) * 1.2
    x = np.linspace(-squrerange, squrerange, 2)
    y = np.linspace(-squrerange, squrerange, 2)
    X, Y = np.meshgrid(x, y)
    for Z in gen_mode["z_range"]:
        Z = np.ones(X.shape) * Z

        # 创建图形

        # 绘制平面
        ax.plot_surface(X, Y, Z, alpha=0.2, rstride=100, cstride=100)

    # 设置轴标签
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()
    pass
