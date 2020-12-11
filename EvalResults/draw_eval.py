from matplotlib import pyplot as plt
import numpy as np

def read_ap(filename):
    cat_cache = ['car_easy', 'car_moderate', 'car_hard', 'pedestrian_easy', 'pedestrian_moderate', 'pedestrian_hard',
                 'cyclist_easy', 'cyclist_moderate', 'cyclist_hard']
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
                iner_count = 0
                continue
            if line == 'Pedestrian AP_R40@0.50, 0.50, 0.50:':
                info_type = 3
                iner_count = 0
                continue
            if line == 'Cyclist AP_R40@0.50, 0.50, 0.50:':
                info_type = 5
                iner_count = 0
                continue
            if info_type == 1 or info_type == 3 or info_type == 5:
                iner_count += 1
            if info_type == 1 and iner_count == 3:
                info = line.split(':')[1].split(',')
                ap_cache['car_easy'].append(float(info[0]))
                ap_cache['car_moderate'].append(float(info[1]))
                ap_cache['car_hard'].append(float(info[2]))
                info_type = 0
                iner_count = 0
                continue
            if info_type == 3 and iner_count == 3:
                info = line.split(':')[1].split(',')
                ap_cache['pedestrian_easy'].append(float(info[0]))
                ap_cache['pedestrian_moderate'].append(float(info[1]))
                ap_cache['pedestrian_hard'].append(float(info[2]))
                info_type = 0
                iner_count = 0
                continue
            if info_type == 5 and iner_count == 3:
                info = line.split(':')[1].split(',')
                ap_cache['cyclist_easy'].append(float(info[0]))
                ap_cache['cyclist_moderate'].append(float(info[1]))
                ap_cache['cyclist_hard'].append(float(info[2]))
                info_type = 0
                iner_count = 0
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
    _, mean_ap_cache['painted_re'], std_ap_cache['painted_re'] = read_ap('pp-gt-v3+-re.txt')
    idx = range(9)
    width = 0.2
    mid_width = 0.01

    plt.figure()
    plt.bar(idx, mean_ap_cache['naive'], width=width, yerr=std_ap_cache['naive'], error_kw={'ecolor': 'r', 'capsize': 3},
            alpha=0.7, label='naive')
    plt.bar([i+(width+mid_width) for i in idx], mean_ap_cache['gt'], width=width, yerr=std_ap_cache['gt'],
            error_kw={'ecolor': 'r', 'capsize': 3}, alpha=0.7, label='GT sampling')
    plt.bar([i+2*(width+mid_width) for i in idx], mean_ap_cache['painted'], width=width, yerr=std_ap_cache['painted'],
            error_kw={'ecolor': 'r', 'capsize': 3}, alpha=0.7, label='GT sampling + Pointpainting')
    plt.bar([i+3*(width+mid_width) for i in idx], mean_ap_cache['painted_re'], width=width, yerr=std_ap_cache['painted_re'],
            error_kw={'ecolor': 'r', 'capsize': 3}, alpha=0.7, label='GT sampling(refined) + Pointpainting')
    plt.xticks([i+(width+mid_width) for i in idx], cat_cache)
    plt.legend()
    plt.title('APs of Each Class Using Different Type of Augmenting Method')
    plt.ylim(40)
    plt.ylabel('AP')

def eval_2d_seg():
    mean_ap_cache = {}
    std_ap_cache = {}
    # TODO: change file names
    cat_cache, mean_ap_cache['dlv3'], std_ap_cache['dlv3'] = read_ap('pp-gt-v3.txt')
    _, mean_ap_cache['dlv3+'], std_ap_cache['dlv3+']= read_ap('pp-gt-v3+.txt')
    _, mean_ap_cache['hma'], std_ap_cache['hma'] = read_ap('pp-gt-hma.txt')
    _, mean_ap_cache['rangenet'], std_ap_cache['rangenet'] = read_ap('pp-gt-r-re.txt')
    idx = range(9)
    width = 0.2
    mid_width = 0.01

    plt.figure()
    plt.bar(idx, mean_ap_cache['dlv3'], width=width, yerr=std_ap_cache['dlv3'], error_kw={'ecolor': 'r', 'capsize': 3},
            alpha=0.7, label='DeepLabv3 on COCO')
    plt.bar([i+(width+mid_width) for i in idx], mean_ap_cache['dlv3+'], width=width, yerr=std_ap_cache['dlv3+'],
            error_kw={'ecolor': 'r', 'capsize': 3}, alpha=0.7, label='DeepLabv3+ on CityScape')
    plt.bar([i+2*(width+mid_width) for i in idx], mean_ap_cache['hma'], width=width, yerr=std_ap_cache['hma'],
            error_kw={'ecolor': 'r', 'capsize': 3}, alpha=0.7, label='Hierarchical Multi-Scale Attentio')
    plt.bar([i+3*(width+mid_width) for i in idx], mean_ap_cache['rangenet'], width=width, yerr=std_ap_cache['rangenet'],
            error_kw={'ecolor': 'r', 'capsize': 3}, alpha=0.7, label='RangeNet')
    plt.xticks([i+(width+mid_width)*3/2 for i in idx], cat_cache)
    plt.legend()
    plt.title('APs of Each Class Using Different 2D Segmentation Networks')
    plt.ylim(40)
    plt.ylabel('AP')

def eval_lidar_net():
    mean_ap_cache = {}
    std_ap_cache = {}
    cat_cache, mean_ap_cache['pointpillars'], std_ap_cache['pointpillars'] = read_ap('pp.txt')
    # TODO: change file names
    _, mean_ap_cache['painted pointpillars'], std_ap_cache['painted pointpillars']= read_ap('pp-gt-v3+-re.txt')
    _, mean_ap_cache['point rcnn'], std_ap_cache['point rcnn'] = read_ap('pr-gt.txt')
    _, mean_ap_cache['painted point rcnn'], std_ap_cache['painted point rcnn'] = read_ap('pr-gt-v3+-re2.txt')
    _, mean_ap_cache['pvrcnn'], std_ap_cache['pvrcnn'] = read_ap('pv.txt')
    _, mean_ap_cache['painted pvrcnn'], std_ap_cache['painted pvrcnn'] = read_ap('pv-gt-v3+-re.txt')
    idx = range(9)
    width = 0.1
    mid_width = 0.01

    plt.figure()
    plt.bar(idx, mean_ap_cache['pointpillars'], width=width, yerr=std_ap_cache['pointpillars'], error_kw={'ecolor': 'r', 'capsize': 3},
            alpha=0.7, label='PointPillars')
    plt.bar([i+(width+mid_width) for i in idx], mean_ap_cache['painted pointpillars'], width=width, yerr=std_ap_cache['painted pointpillars'],
            error_kw={'ecolor': 'r', 'capsize': 3}, alpha=0.7, label='painted pointpillars')
    plt.bar([i+2*(width+mid_width) for i in idx], mean_ap_cache['point rcnn'], width=width, yerr=std_ap_cache['point rcnn'],
            error_kw={'ecolor': 'r', 'capsize': 3}, alpha=0.7, label='point rcnn')
    plt.bar([i+3*(width+mid_width) for i in idx], mean_ap_cache['painted point rcnn'], width=width, yerr=std_ap_cache['painted point rcnn'],
            error_kw={'ecolor': 'r', 'capsize': 3}, alpha=0.7, label='painted point rcnn')
    plt.bar([i+4*(width+mid_width) for i in idx], mean_ap_cache['pvrcnn'], width=width, yerr=std_ap_cache['pvrcnn'],
            error_kw={'ecolor': 'r', 'capsize': 3}, alpha=0.7, label='pvrcnn')
    plt.bar([i+5*(width+mid_width) for i in idx], mean_ap_cache['painted pvrcnn'], width=width, yerr=std_ap_cache['painted pvrcnn'],
            error_kw={'ecolor': 'r', 'capsize': 3}, alpha=0.7, label='painted pvrcnn')
    plt.xticks([i+(width+mid_width) for i in idx], cat_cache)
    plt.legend()
    plt.title('APs of Each Class Using Different Lidar Detection Networks')
    plt.ylim(40)
    plt.ylabel('AP')

def plots():
    eval_augmentor()
    eval_2d_seg()
    eval_lidar_net()
    plt.show()

if __name__ == "__main__":
    plots()
