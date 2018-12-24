import dataloader
from utils import *


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'label'])
        writer.writerows(results)



if __name__ == '__main__':

    rgb_preds = 'record/spatial/spatial_video_preds.pickle'
    opf_preds = 'record/motion/motion_video_preds.pickle'

    with open(rgb_preds,'rb') as f:
        rgb =pickle.load(f)
    f.close()
    with open(opf_preds,'rb') as f:
        opf =pickle.load(f)
    f.close()

    dataloader = dataloader.spatial_dataloader(BATCH_SIZE=1, num_workers=1,
                                               path='/home/lb/video_action_recognition/data/jpegs_256',
                                               ucf_list='/home/lb/video_action_recognition/data/datalists',
                                               ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()

    video_level_preds = np.zeros((len(list(rgb.keys())),101))
    video_level_labels = np.zeros(len(list(rgb.keys())))
    correct=0
    ii=0

    final_result = []

    for name in sorted(rgb.keys()):   
        r = rgb[name]
        o = opf[name]

        label = int(test_video[name])-1
                    
        video_level_preds[ii,:] = (r+o)
        video_level_labels[ii] = label
        ii+=1         
        if np.argmax(r+o) == (label):
            correct+=1

        file = "{}/v_{}.avi".format(name.split("_")[0], name)
        pred = np.argmax(r + o) + 1

        final_result.append((file, pred))

        if pred == label:
            correct += 1

    write_csv(final_result, "./submission_fusion.csv")

    # video_level_labels = torch.from_numpy(video_level_labels).long()
    # video_level_preds = torch.from_numpy(video_level_preds).float()
    #
    # top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))
    #
    # print(top1,top5)
    #
    # final_result = []
    #
    # for name in self.dic_video_level_preds.keys():
    #
    #     preds = self.dic_video_level_preds[name]
    #     label = int(self.test_video[name])
    #
    #     video_level_preds[ii, :] = preds
    #     video_level_labels[ii] = label
    #     ii += 1
    #     if np.argmax(preds) == (label):
    #         correct += 1
    #
    #
    #
    # write_csv(final_result, "./submission_motion.csv")
    print("acc : {:.3f} %".format(correct / len(list(rgb.keys())) * 100))
