from matplotlib import pyplot as plt
import numpy as np

def read_ap(filename):
    # cat_cache = ['car_easy', 'car_moderate', 'car_hard', 'pedestrian_easy', 'pedestrian_moderate', 'pedestrian_hard',
    #              'cyclist_easy', 'cyclist_moderate', 'cyclist_hard']
    cat_cache = ['V_e', 'V_m', 'V_h', 'P_e', 'P_m', 'P_h', 'C_e', 'C_m', 'C_h']
    ap_cache = {}
    for cat in cat_cache:
        ap_cache[cat] = []
    mean_ap_cache = []
    std_ap_cache = []
    info_type = 0
    reader = open(filename)
    while True:
        lines = reader.readlines(10000)
        if not lines:
            break
        for line in lines:
            line = line.strip('\n')
            if line == 'Car AP_R40@0.70, 0.70, 0.70:':
                info_type = 1
                continue
            if line == 'Pedestrian AP_R40@0.50, 0.50, 0.50:':
                info_type = 3
                continue
            if line == 'Cyclist AP_R40@0.50, 0.50, 0.50:':
                info_type = 5
                continue
            if info_type == 1 or info_type == 3 or info_type == 5:
                info_type += 1
                continue
            if info_type == 2:
                info = line.split(':')[1].split(',')
                # ap_cache['car_easy'].append(float(info[0]))
                # ap_cache['car_moderate'].append(float(info[1]))
                # ap_cache['car_hard'].append(float(info[2]))
                ap_cache['V_e'].append(float(info[0]))
                ap_cache['V_m'].append(float(info[1]))
                ap_cache['V_h'].append(float(info[2]))
                info_type = 0
                continue
            if info_type == 4:
                info = line.split(':')[1].split(',')
                # ap_cache['pedestrian_easy'].append(float(info[0]))
                # ap_cache['pedestrian_moderate'].append(float(info[1]))
                # ap_cache['pedestrian_hard'].append(float(info[2]))
                ap_cache['P_e'].append(float(info[0]))
                ap_cache['P_m'].append(float(info[1]))
                ap_cache['P_h'].append(float(info[2]))
                info_type = 0
                continue
            if info_type == 6:
                info = line.split(':')[1].split(',')
                # ap_cache['cyclist_easy'].append(float(info[0]))
                # ap_cache['cyclist_moderate'].append(float(info[1]))
                # ap_cache['cyclist_hard'].append(float(info[2]))
                ap_cache['C_e'].append(float(info[0]))
                ap_cache['C_m'].append(float(info[1]))
                ap_cache['C_h'].append(float(info[2]))
                info_type = 0
                continue
    for cat in cat_cache:
        mean_ap_cache.append(np.mean(np.array(ap_cache[cat])))
        std_ap_cache.append(np.std(np.array(ap_cache[cat])))
    reader.close()
    return (cat_cache, mean_ap_cache, std_ap_cache)

def eval_augmentor():
    mean_ap_cache = {}
    std_ap_cache = {}
    cat_cache, mean_ap_cache['naive'], std_ap_cache['naive'] = read_ap('pp.txt')
    # TODO: change file names
    _, mean_ap_cache['gt'], std_ap_cache['gt']= read_ap('pp-gt.txt')
    _, mean_ap_cache['painted'], std_ap_cache['painted'] = read_ap('pp-gt-v3+.txt')
    _, mean_ap_cache['rec'], std_ap_cache['rec'] = read_ap('pp-gt-v3+-re.txt')
    idx = range(9)
    width = 0.2
    mid_width = 0.01

    plt.figure()
    plt.bar(idx, mean_ap_cache['naive'], width=width, yerr=std_ap_cache['naive'], error_kw={'ecolor': 'gray', 'capsize': 1.5},
            alpha=0.7, label='Naive')
    plt.bar([i+(width+mid_width) for i in idx], mean_ap_cache['gt'], width=width, yerr=std_ap_cache['gt'],
            error_kw={'ecolor': 'gray', 'capsize': 1.5}, alpha=0.7, label='GT sampling')
    plt.bar([i+2*(width+mid_width) for i in idx], mean_ap_cache['painted'], width=width, yerr=std_ap_cache['painted'],
            error_kw={'ecolor': 'gray', 'capsize': 1.5}, alpha=0.7, label='GT sampling + Pointpainting')
    plt.bar([i+3*(width+mid_width) for i in idx], mean_ap_cache['rec'], width=width, yerr=std_ap_cache['rec'],
            error_kw={'ecolor': 'gray', 'capsize': 1.5}, alpha=0.7, label='Rectified GT sampling + Pointpainting')
    plt.xticks([i+3*(width+mid_width)/2 for i in idx], cat_cache)
    plt.legend()
    plt.title('APs of Each Class Using Different Segmentation Guidance')
    plt.ylim(40)
    plt.ylabel('Preformance(AP)')

def eval_2d_seg():
    mean_ap_cache = {}
    std_ap_cache = {}
    # TODO: change file names
    cat_cache, mean_ap_cache['dlv3'], std_ap_cache['dlv3'] = read_ap('pp-gt-v3.txt')
    _, mean_ap_cache['dlv3+'], std_ap_cache['dlv3+']= read_ap('pp-gt-v3+.txt')
    _, mean_ap_cache['hma'], std_ap_cache['hma'] = read_ap('pp-gt-hma.txt')
    _, mean_ap_cache['rangenet'], std_ap_cache['rangenet'] = read_ap('pp-gt-r-re.txt')
    _, mean_ap_cache['sq'], std_ap_cache['sq'] = read_ap('pp-gt-sq-re.txt')
    idx = range(9)
    width = 0.15
    mid_width = 0.01

    plt.figure()
    plt.bar(idx, mean_ap_cache['dlv3'], width=width, yerr=std_ap_cache['dlv3'], error_kw={'ecolor': 'gray', 'capsize': 1.5},
            alpha=0.7, label='DeepLabv3 on COCO')
    plt.bar([i+(width+mid_width) for i in idx], mean_ap_cache['dlv3+'], width=width, yerr=std_ap_cache['dlv3+'],
            error_kw={'ecolor': 'gray', 'capsize': 1.5}, alpha=0.7, label='DeepLabv3+ on CityScape')
    plt.bar([i+2*(width+mid_width) for i in idx], mean_ap_cache['hma'], width=width, yerr=std_ap_cache['hma'],
            error_kw={'ecolor': 'gray', 'capsize': 1.5}, alpha=0.7, label='Hierarchical Multi-Scale Attention')
    plt.bar([i+3*(width+mid_width) for i in idx], mean_ap_cache['rangenet'], width=width, yerr=std_ap_cache['rangenet'],
            error_kw={'ecolor': 'gray', 'capsize': 1.5}, alpha=0.7, label='SPVNAS')
    plt.bar([i+4*(width+mid_width) for i in idx], mean_ap_cache['sq'], width=width, yerr=std_ap_cache['sq'],
            error_kw={'ecolor': 'gray', 'capsize': 1.5}, alpha=0.7, label='SqueezeSegV3')
    plt.xticks([i+2*(width+mid_width) for i in idx], cat_cache)
    plt.legend()
    plt.title('APs of Each Class Using Different Segmentation Networks')
    plt.ylim(40)
    plt.ylabel('Performance(AP)')

def eval_lidar_net():
    mean_ap_cache = {}
    # TODO: change file names
    _, mean_ap_cache['pp'], _ = read_ap('pp.txt')
    _, mean_ap_cache['pr'], _= read_ap('pr.txt')
    _, mean_ap_cache['pv'], _ = read_ap('pv.txt')
    _, mean_ap_cache['ppgt'], _ = read_ap('pp-gt.txt')
    _, mean_ap_cache['prgt'], _= read_ap('pr-gt.txt')
    _, mean_ap_cache['pvgt'], _ = read_ap('pv-gt.txt')
    _, mean_ap_cache['ppre'], _ = read_ap('pp-gt-v3+-re.txt')
    _, mean_ap_cache['prre'], _= read_ap('pr-gt-v3+-re.txt')
    _, mean_ap_cache['pvre'], _ = read_ap('pv-gt-v3+-re.txt')
    map_naive, map_gt, map_re = [], [], []
    for method in ['pp', 'pr', 'pv']:
        map_naive.append(np.mean(np.array(mean_ap_cache[method])))
        map_gt.append(np.mean(np.array(mean_ap_cache[method+'gt'])))
        map_re.append(np.mean(np.array(mean_ap_cache[method+'re'])))
    idx = range(3)
    width = 0.24
    mid_width = 0.01

    plt.figure()
    plt.bar(idx, map_naive, width=width, error_kw={'ecolor': 'gray', 'capsize': 1.5},
            alpha=0.7, label='naive')
    plt.bar([i+(width+mid_width) for i in idx], map_gt, width=width,
            error_kw={'ecolor': 'gray', 'capsize': 1.5}, alpha=0.7, label='GT sampling')
    plt.bar([i+2*(width+mid_width) for i in idx], map_re, width=width,
            error_kw={'ecolor': 'gray', 'capsize': 1.5}, alpha=0.7, label='Rectified GT sampling + Pointpainting')
    plt.xticks([i+(width+mid_width) for i in idx], ['PointPillars', 'PointRCNN', 'PV-RCNN'])
    plt.legend()
    plt.title('mAPs through All Classes Using Different Lidar Detection Networks')
    plt.ylim(60)
    plt.ylabel('Performance(mAP)')

def plots():
    eval_augmentor()
    eval_2d_seg()
    eval_lidar_net()
    plt.show()

if __name__ == "__main__":
    plots()
