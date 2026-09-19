from tshub.utils.plot_reward_curves import plot_reward_curve
from tshub.utils.get_abs_path import get_abs_path
path_convert = get_abs_path(__file__)


if __name__ == '__main__':
    log_files = [
        path_convert(f'./log/{i}.monitor.csv')
        for i in range(4)
    ]
    print(log_files)
    plot_reward_curve(log_files, output_file ='./log/',window_size =10, fill_outliers = False)