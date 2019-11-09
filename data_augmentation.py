import imgaug as ia
import argparse
from imgaug import augmenters as iaa
from shapely.geometry import Polygon
from cvat import CvatDataset
import shapely
import numpy as np
from urllib.request import urlopen
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw
import random
import garbage as g
from tqdm import tqdm

import albumentations as albu
import numpy as np
import cv2
import os

num = 0
width_of_new_image = 0
height_of_new_image = 0
image_id_global = 0

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_image(data, file_extension, image_name) :
    global num
    data = (data * 255).astype('uint8')
    image_name = image_name[:image_name.rfind('.')]
    im = Image.fromarray(data)
    _output_folder = args.output_folder
    if not(os.path.exists(_output_folder)):
        path = ''
        for i, _ in enumerate(_output_folder.split('/')):
            path += _output_folder.split('/')[i] + '/'
            if not(os.path.exists(path)): os.mkdir(path)
    im.save(_output_folder+'/'+image_name + "_" + str(num).zfill(4)+file_extension)
    
############################################################################################################

def points_please():
    
    # output: list of dicts with size - (number of founded polygons and boxes). Each record in dict contains 'image_id',
    # 'polygon', 'label', 'occluded' and 'conf'
    # example:
    # list_of_polygons = [
    #    {image_id: 0, 
    #    'polygon': [ (0,0), (1,1), (1,0) ],
    #    'occluded': 0,
    #    'label': 'person'
    #    'conf': NA
    #    }
    #    {....
    #    }
    # ]
    # Внимание !!! Image_id - это старые id картинок, по которым делаем потом аугментацию. В выходной файл эти image_id 
    # выводить нельзя
    
    global input_CvatDataset
    global output_Cvat_Dataset
    
    list_of_polygons = []
    for _image_id in input_CvatDataset.get_image_ids():
        
        for polygon in input_CvatDataset.get_polygons(_image_id):
            list_of_polygons.append({})
            list_of_polygons[-1]['image_id'] = _image_id
            list_of_polygons[-1]['polygon'] = [tuple(x_y) for x_y in polygon['points']]
            list_of_polygons[-1]['label'] = polygon['label']
            list_of_polygons[-1]['occluded'] = polygon['occluded']
            list_of_polygons[-1]['conf'] = polygon['conf']
            list_of_polygons[-1]['name'] = input_CvatDataset.get_name(_image_id).split('/')[-1]
 
        for box in input_CvatDataset.get_boxes(_image_id):
            xtl, ytl, xbr, ybr = box['xtl'], box['ytl'], box['xbr'], box['ybr']
            polygon = [ (xtl,ytl), (xbr,ytl), (xbr, ybr), (xtl,ybr) ]
            list_of_polygons.append({})
            list_of_polygons[-1]['image_id'] = _image_id
            list_of_polygons[-1]['polygon'] = polygon
            list_of_polygons[-1]['label'] = box['label']
            list_of_polygons[-1]['occluded'] = box['occluded']
            list_of_polygons[-1]['conf'] = box['conf']
            list_of_polygons[-1]['name'] = input_CvatDataset.get_name(_image_id).split('/')[-1]
        
    return list_of_polygons

#======================================================================================

def pol_to_dots(poly):
    l = list()
    x, y = poly.exterior.coords.xy
    coord = [x, y]
    for i in range(len(coord[0])):
        xy = [coord[0][i], coord[1][i]]
        dots = tuple(xy)
        l.append(dots)

    return l

def checked(pol):
    global height_of_new_image
    global width_of_new_image
    
    pol = Polygon(pol)
    p1 = Polygon([(0, 0), (0, height_of_new_image), (width_of_new_image, height_of_new_image), (width_of_new_image, 0)])
    try:
        p1 = p1.intersection(pol)
    except:
        return 0
    if type(p1) == shapely.geometry.multipolygon.MultiPolygon:
        return 0
    if p1.is_empty == True:
        return 0
    else: 
        #print(p1)
        #print('height_of_new_image = ' + str(height_of_new_image))
        #print('width_of_new_image = ' + str(width_of_new_image))
        return pol_to_dots(p1)


def augment_please(image, polygons_of_image, image_name):
    global output_CvatDataset
    global num
    global width_of_new_image
    global height_of_new_image
    global image_id_global
    
    num_of_augmentations = args.num_output_images_from_one_image
    sometimes = lambda aug: iaa.Sometimes(0.5, aug) # С вероятностью 0.5 будет делать перспективную трансформацию, см. ниже Perspectiv
    full_list = list()
    num = 0
    for i in range(num_of_augmentations):
        if args.random_sizes == False:
            width_of_new_image = args.max_width
            height_of_new_image = args.max_height
        else:
            if args.preserve_ratio_of_sides == False:
                width_of_new_image = random.randint(args.min_width, args.max_width)
                height_of_image = random.randint(args.min_height, args.max_height)
            else:
                k_height = args.max_height / args.min_height
                k_width = args.max_width / args.min_width
                max_k = k_height
                if k_width < max_k: max_k = k_width
                k = random.uniform(1.0, max_k)
                width_of_new_image = int(args.min_width*k)
                height_of_new_image = int(args.min_height*k)
        aug = iaa.Sequential([
            iaa.Affine(rotate=(-10, 10)),
            iaa.CropToFixedSize(width=width_of_new_image, height=height_of_new_image),
            iaa.Fliplr(args.flip_horizontal_probab),
            iaa.Flipud(args.flip_vertical_probab),
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.10)))
        ])
        aug = aug.to_deterministic()
        batch_aug = aug.augment(
            images=[image], polygons=[[_polygon['polygon'] for _polygon in polygons_of_image]],
            return_batch=True)

        image_augmentated = batch_aug.images_aug
        image_augmentated = np.asarray(image_augmentated)[0]

        polygons_augmentated = batch_aug.polygons_aug
        polygons_augmentated = np.asarray(polygons_augmentated)
        
        new_polygons_intersected = []
        for p in polygons_augmentated:
            for j, polygon in enumerate(p):
                if checked(polygon) == 0:
                    continue
                else:
                    new_polygons_intersected.append({})
                    new_polygons_intersected[-1]['polygon'] = [ [x[0], x[1]] for x in checked(polygon) ]
                    new_polygons_intersected[-1]['label'] = polygons_of_image[j]['label']
                    new_polygons_intersected[-1]['conf'] = polygons_of_image[j]['conf']
            
        save_image(image_augmentated, ".jpg", image_name)
           
        output_CvatDataset.add_image(image_id_global)
        output_CvatDataset.set_name(image_id_global, image_name[:image_name.rfind('.')] + "_" + str(num).zfill(4)+'.jpg')
        output_CvatDataset.set_size(image_id_global, len(image_augmentated[0]), len(image_augmentated))
            
        if len(new_polygons_intersected) != 0:
            for k, _record in enumerate(new_polygons_intersected):
                output_CvatDataset.add_polygon(
                    image_id=image_id_global,
                    points=_record['polygon'],
                    label=_record['label'],
                    conf=_record['conf'],
                    occluded=0
                )
        num += 1
        image_id_global += 1
    return None



def get_polygons(list_of_polygons):

    for filename in tqdm(sorted(os.listdir(args.initial_folder))):
        image = mpimg.imread(args.initial_folder+'/'+filename)
        polygons_of_image = [x for x in list_of_polygons if x['name'] == filename]
        augment_please(image, polygons_of_image, filename)

    return None


def build_parser():
    parser = argparse.ArgumentParser("Add polygons according to sign class")
    
    parser.add_argument(
        "--input-file",
        default='cds-data-preparation-master/63_ivb_nkbvs_selected_joined_05_11_19_15_49.xml',
        type=str,
        help='File of type .xml that contains annotations (for example, from CVAT)'        
    )
    parser.add_argument(
        "--output-file",
        default='cds-data-preparation-master/results/test_set/annotations_test_set.xml',
        type=str,
        help='File of type .xml that will contain resulting information'
    )
    parser.add_argument(
        "--initial-folder",
        default='yolact/data/test_set',
        type=str,
        help='Directory that contains images to augmentate'
    )
    parser.add_argument(
        "--output-folder",
        default='cds-data-preparation-master/results/test_set/images',
        type=str,
        help='Output directory that will contain augmentated imaged'
    )
    
    parser.add_argument(
        "--flip-horizontal-probab",
        default=0.5,
        type=float,
        help='Flip/mirror input images horizontally (default=0.5)'
    )
    
    parser.add_argument(
        "--flip-vertical-probab",
        default=0.0,
        type=float,
        help='Flip/mirror input images vertical (default=0)'
    )
    
    parser.add_argument(
        "--max-height",
        default=1280,
        type=int,
        help='Max height of the output augmentated image (default=1280)'
    )
    
    parser.add_argument(
        "--max-width",
        default=2048,
        type=int,
        help='Max width of the output augmentated image (default=2048)'
    )
    
    parser.add_argument(
        "--min-height",
        default=384,
        type=int,
        help='Min height of the output augmentated image (default=384)'
    )
    
    parser.add_argument(
        "--min-width",
        default=640,
        type=int,
        help='Min width of the output augmentated image (default=640)'
    )
    
    
    parser.add_argument(
        "--preserve-ratio-of-sides",
        default=True,
        type=str2bool,
        help='True/False. If output images should preserve ratio of the input image'
    )    
    
    parser.add_argument(
        "--random-sizes",
        default=True,
        type=str2bool,
        help='True/False. If output images should have random sizes from min (height and width) to max (height and width)\n'+
            'If random-sizes = False then augmentated images will have size - (max-height, max-width), look for these parsed arguments'
    )

    parser.add_argument(
        "--num-output-images-from-one-image",
        default=18,
        type=int,
        help='Number of augmentated images made from one image (default=16)'
    )
    
    return parser

def dump(output_file):
    global output_CvatDataset
                                        
    output_CvatDataset.dump(output_file)


def main(args):
    global output_CvatDataset
    global input_CvatDataset
    
    input_CvatDataset = CvatDataset()
    input_CvatDataset.load(args.input_file)
    output_CvatDataset = CvatDataset()
    
    list_of_polygons = points_please()
    get_polygons(list_of_polygons)
    dump(args.output_file)

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
