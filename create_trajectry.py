
import os
import glob
import argparse
import pickle

import numpy as np
import cv2
from PIL import Image, ImageDraw


side_map = {0:'Left', 1:'Right'}
state_map2 = {0:'N', 1:'S', 2:'O', 3:'P', 4:'F'}
hand_rgb = [(0, 90, 181), (220, 50, 32)]
hand_rgba = [(0, 90, 181, 70), (220, 50, 32, 70)]


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument("--hod_path", help="Path to detections.", type=str, required=True)
    parser.add_argument("--dataset_path", help="Path to image dataset.", type=str, required=True)
    parser.add_argument("--save_path", help="Path to saved image.", type=str, required=True)
    args = parser.parse_args()
    return args


def tracking(hod_data):
    l_traj = []
    r_traj = []
    frame_indice = []
    for item in hod_data:
        left = np.full((10), -1)
        right = np.full((10), -1)
        for det in item['hand_dets']:
            if side_map[det[9]] == 'Left':
                left = det
            elif side_map[det[9]] == 'Right':
                right = det
        l_traj.append(left)
        r_traj.append(right)
        frame_indice.append(item['frame_index'])
    l_traj = np.array(l_traj)
    r_traj = np.array(r_traj)
    return l_traj, r_traj, frame_indice


def vis_traj(l_traj, r_traj, frame_indice, basename, framedir):
    savedir = os.path.join(args.save_path, basename)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    for i, frame in enumerate(frame_indice):
        im_file = os.path.join(framedir, frame)
        im = cv2.imread(im_file)
        im = im[:,:,::-1]
        image = Image.fromarray(im).convert("RGBA")
        width, height = image.size

        for k, det in enumerate([l_traj[i], r_traj[i]]):
            if det[9] != -1:
                # set
                bbox = list(int(np.round(x)) for x in det[:4])
                cie_x = int(np.round((det[0]+det[2])/2))
                cie_y = int(np.round((det[1]+det[3])/2))
                cie = [cie_x, cie_y]
                score = det[4]
                lr = det[9]
                state = det[5]
                # write
                mask = Image.new('RGBA', (width, height))
                pmask = ImageDraw.Draw(mask)
                pmask.ellipse((cie_x-3, cie_y-3, cie_x+3, cie_y+3), fill=hand_rgb[k], outline=hand_rgb[k])
                #pmask.rectangle(bbox, outline=hand_rgb[k], width=4, fill=hand_rgba[k])
                image.paste(mask, (0,0), mask)

        # save
        result_name = '{}_det.png'.format(frame.replace('.png', ''))
        result_path = os.path.join(savedir, result_name)
        image.save(result_path)


if __name__ == '__main__':
    args = parse_args()

    patt1 = os.path.join(args.hod_path, 'P*')
    for box_dir in glob.glob(patt1):
        patt2 = os.path.join(box_dir, '*.pkl')
        for box_pkl in glob.glob(patt2):
            with open(box_pkl, 'rb') as f:
                hod_data = pickle.load(f)
                basename = os.path.basename(box_pkl).replace('.pkl', '')
                framedir = os.path.join(args.dataset_path, basename[0:3], 'rgb_frames', basename[0:6])
                print('Processing {} ...'.format(basename))
                l_traj, r_traj, frame_indice = tracking(hod_data)
                vis_traj(l_traj, r_traj, frame_indice, basename, framedir)
