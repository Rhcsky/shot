import hydra

from configs.configs import Config
from dataloader import get_source_dataloader, get_target_dataloader
from ori_dloader import data_load_source, data_load_target
from train_source import init_seed, init_dataset_name

cfg = None


@hydra.main(config_path='configs', config_name='config')
def main(cfg: Config) -> None:
    init_seed(cfg.train.seed)
    init_dataset_name(cfg)

    my_train_loader, my_valid_loader = get_source_dataloader(cfg)
    my_test_loader_list = get_target_dataloader(cfg)

    train_loader, valid_loader = data_load_source(cfg)
    test_loader_list = data_load_target(cfg)

    # print(len(my_test_loader_list[0]))
    # print(len(my_test_loader_list[1]))
    # print(len(my_test_loader_list[2]))
    # print(len(test_loader_list[0]))
    # print(len(test_loader_list[1]))
    # print(len(test_loader_list[2]))
    # print(my_test_loader_list[0] == test_loader_list[0])
    # src_classes = [i for i in range(65)]
    #
    # all_txt_src = ImageFolder('../../data/office-home/Art', loader=lambda x: x)
    # txt_src = list()
    # for x in all_txt_src:
    #     path = x[0]
    #     cls = x[1]
    #     txt_src.append(f"{path} {cls}")
    #
    # if not cfg.da.type == 'uda':
    #     label_map_s = {}
    #     for i in range(65):
    #         label_map_s[src_classes[i]] = i
    #
    #     new_src = []
    #     for i in range(len(txt_src)):
    #         rec = txt_src[i]
    #         reci = rec.strip().split(' ')
    #         if int(reci[1]) in src_classes:
    #             line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
    #             new_src.append(line)
    #     txt_src = new_src.copy()
    #
    # ds = ImageList(txt_src, transform=image_train())
    #
    # train_transform = transforms.Compose([
    #     # transforms.Resize((cfg.dataset.resize_size, cfg.dataset.resize_size)),
    #     # transforms.RandomCrop(cfg.dataset.crop_size),
    #     # transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    # folder_dataset = ImageFolder('../../data/office-home/Art', loader=lambda x: x)
    # mds = ImageDataset(folder_dataset, True, 65, 25, train_transform)
    #
    # print(len(ds))
    # print(len(mds))
    #
    # xx, yy = mds.x, mds.y
    #
    # for imgs, x, y in zip(ds.imgs, xx, yy):
    #     ax, ay = imgs[0], imgs[1]
    #
    #     if ax == x and ay == y:
    #         continue
    #     else:
    #         print("HEHE")
    #
    # print(ds[0][0] == mds[0][0])


if __name__ == '__main__':
    main()
