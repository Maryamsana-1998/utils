import io
import itertools
from bbox_overlaps import bbox_overlaps, bbox_overlaps_as_percentage_of_first_bbox
from typing import Dict, Sequence, Any,List, Tuple
from numbers import Number
import numpy as np
import pandas as pd
import fitz
from lxml import etree
from PIL import Image

def combine_bboxes(bboxes):
    if len(bboxes) == 1:
        return bboxes[0]
    bboxes = np.array(bboxes)
    x1, y1, x2, y2 = [bboxes[:,i] for i in range(4)]
    nx1, ny1 = [a.min() for a in (x1, y1)]
    nx2, ny2 = [a.max() for a in (x2, y2)]
    return [nx1,ny1,nx2,ny2]

def transfer_bbox_labels(labels:Sequence[Any],labeled_bboxes:Sequence[Sequence[Number]],unlabeled_bboxes:Sequence[Sequence[Number]]) -> List[Any]:
    """
    Args:
        labels (Sequence[Any]): Labels for the labeled_bboxes
        labeled_bboxes (Sequence[Sequence[Number]]): The bboxes, with known labels
        unlabeled_bboxes (Sequence[Sequence[Number]]): The bboxes in the same image, with unknown labels.

    Returns:
        List[Any]: A list of the same length as unlabeled_bboxes, giving the corresponding labels
    """
    labeled_bboxes = np.array(labeled_bboxes)
    unlabeled_bboxes = np.array(unlabeled_bboxes)
    if len(labeled_bboxes) != 0:
        matches = (bbox_overlaps_as_percentage_of_first_bbox( labeled_bboxes, unlabeled_bboxes ) > 0.6).T.astype(np.float)
        no_match_rows = np.logical_not(matches.sum(axis=1) > 0)
        try:
            match_indices = matches.argmax( axis=1 )
        except ValueError:
            match_indices = np.ones(((unlabeled_bboxes.shape[0]),1)) * -1
        match_indices[no_match_rows] = -1

        new_labels = []
        
        for match_index in match_indices:
            if match_index == -1:
                new_labels.append(None)
            else:
                new_labels.append(labels[match_index])
        assert len(new_labels) == unlabeled_bboxes.shape[0]
    else:
        new_labels = [None]*len(unlabeled_bboxes)
    return new_labels

def get_pdf_ocr(pdf_path, page_idx, remove_impossible_boxes=True):
    idx = page_idx
    doc = fitz.open(str(pdf_path))  # open pdf files using fitz bindings
    page = doc[idx]

    flags = fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE
#    with open('txtfile.txt','rb') as f:
 #       txt = f.read()

    # blocks = page.get_text("rawdict", flags=flags)["blocks"]

    txt = page.get_text('words', flags=flags)
    df = pd.DataFrame(txt,columns=['x0','y0','x1','y1','words','block_no', 'line_no', 'word_no'])
    h, w = [int(page.get_text('rawdict')[x]) for x in ['height', 'width']]
    if remove_impossible_boxes:
        df = df[~(df['x0'] < 0)]
        df = df[~(df['y0'] < 0)]
        df = df[~(df['x1'] > w)]
        df = df[~(df['y1'] > h)]
    ocr_data = (df.to_dict(orient='list'))
    x0, y0, x1, y1 = [ocr_data[x] for x in ['x0','y0','x1','y1']]
    bboxes = [*zip(x0, y0, x1, y1)]
    for x in ['x0','y0','x1','y1']:
        del ocr_data[x]
    ocr_data['bboxes'] = bboxes
    ocr_data['size'] = (w,h)

    return ocr_data

def get_data_from_xml(fpath):
    annotation_labels = []
    annotation_bboxes = []
    with open(fpath,'rb') as f:
        xmlstring = f.read()
        annotation_tag = etree.fromstring(xmlstring)
        size_tag = annotation_tag.findall('size')[0]
        width = int(size_tag.findall('width')[0].text)
        height = int(size_tag.findall('height')[0].text)
        object_tags = annotation_tag.findall('object')
        for object_tag in object_tags:
            name = object_tag.find('name').text
            bndbox = object_tag.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)
            bbox = [xmin,ymin,xmax,ymax]
            annotation_labels.append(name)
            annotation_bboxes.append(bbox)
    return {
        'annotation_bboxes' : annotation_bboxes,
        'annotation_labels' : annotation_labels,
        'annotation_size'   : (width, height)
    }

def resize_bboxes_according_to_new_image_size(bboxes1, size1, size2):
    b1 = np.array(bboxes1)
    ow, oh = size1
    nw, nh = size2
    b1[:,[0,2]] *= nw/ow
    b1[:,[1,3]] *= nh/oh
    return b1.tolist()

def combine_ocr_data_and_ann_data(ocr_data, ann_data, filter_impossible_boxes = True):
    annotation_labels = ann_data['annotation_labels']
    annotation_bboxes = ann_data['annotation_bboxes']
    bboxes = ocr_data['bboxes']
    annotation_size = ann_data['annotation_size']
    ocr_size = ocr_data['size']
    resized_bboxes = resize_bboxes_according_to_new_image_size(bboxes, ocr_size, annotation_size)
    annotation_labels,annotation_bboxes = [*zip(*[(l,b) for l,b in zip(annotation_labels,annotation_bboxes) if l == 'item'])]
    labels = transfer_bbox_labels(annotation_labels,annotation_bboxes,resized_bboxes)
    labels = [x if not x is None else 'O' for x in labels]
    annotation_block_label_original = [*zip(*enumerate(annotation_bboxes,start=1))][0]
    annotation_block_label_word_level = transfer_bbox_labels(annotation_block_label_original, annotation_bboxes, resized_bboxes)
    annotation_block_label_word_level = [x if not x is None else -1 for x in annotation_block_label_word_level]
    iob_labels = [[0,0,0,0,0,0] for x in labels]
    for unq in set(annotation_block_label_word_level):
        if unq != -1:
            compliant_idxs = [i for i,x in enumerate(annotation_block_label_word_level) if x == unq]
            
            left_word_i = sorted(compliant_idxs,key=lambda x:resized_bboxes[x][0])[0]
            top_word_i = sorted(compliant_idxs,key=lambda x:resized_bboxes[x][1])[0]
            right_word_i = sorted(compliant_idxs,key=lambda x:-resized_bboxes[x][2])[0]
            bottom_word_i = sorted(compliant_idxs,key=lambda x:-resized_bboxes[x][3])[0]

            iob_labels[left_word_i][0] = 1 #'left_item'
            iob_labels[top_word_i][1] = 1 #'top_item'
            iob_labels[right_word_i][2] = 1 #'right_item'
            iob_labels[bottom_word_i][3] = 1 #'bottom_item'

            for compliant_idx in compliant_idxs:
                if not compliant_idx in set([left_word_i,top_word_i,right_word_i,bottom_word_i]):
                    iob_labels[compliant_idx][4] = 1 #'inside_item'
    for i,iob_label, old_label in zip(itertools.count(), iob_labels, labels):
        if all([x == 0 for x in iob_label[:5]]):
            iob_labels[i][5] = 1
    
    boundaries = [set() for x in labels]
    for i, abl, label in [*zip(itertools.count(), annotation_block_label_word_level, labels)][1:]:
        prev_abl = annotation_block_label_word_level[i-1]
        prev_label = labels[i-1]
        if prev_abl != abl or prev_label != label:
            boundaries[i].add('B')
            boundaries[i-1].add('L')
        if prev_label == label:
            boundaries[i].add('I')
    
    for i in range(len(boundaries)):
        if set(['B','L']) <= boundaries[i]:
            boundaries[i] = set('U')
        if set(['I','L']) <= boundaries[i]:
            boundaries[i] = set('L')
        if set(['I','B']) <= boundaries[i]:
            boundaries[i] = set('B')
        if labels[i] == 'O':
            boundaries[i] = set('O')
        assert len(boundaries[i]) == 1
    boundaries = [list(x)[0] for x in boundaries]
    bilou_labels = [f"{bo}-{lb}" if not lb == 'O' else 'O' for lb,bo in zip(labels,boundaries)]


    assert all([x != 'item' for x in iob_labels])        

    finaldata = {}
    finaldata.update(ocr_data)
    finaldata.update(ann_data)
    finaldata['bboxes'] = resized_bboxes
    finaldata['size']   = annotation_size
    finaldata['labels'] = labels
    finaldata['annotation_block_label'] = annotation_block_label_word_level
    finaldata['iob_labels'] = iob_labels
    finaldata['bilou_labels'] = bilou_labels
    return finaldata

def get_pdf_pix(pdf_path, idx):
    doc = fitz.open(str(pdf_path))  # open pdf files using fitz bindings
    page = doc[idx]
    pix = page.get_pixmap(dpi=300)  # render page to an image
    return pix

def get_pdf_data(pdf_path, idx):
    ocr_data, pix = get_pdf_ocr(pdf_path, idx), get_pdf_pix(pdf_path, idx)
    data = pix.tobytes("png")
    image = Image.open(io.BytesIO(data))
    return ocr_data, image

def get_tight_boxes_to_imgj_data_item(data_item):
    unique_labels = set(data_item['annotation_block_label'])

    sequences_found = []
    for unique_label in unique_labels:
        sequences_found.append([i for i,x in enumerate(data_item['annotation_block_label']) if x == unique_label])

    entity_boxes = []

    for seq in sequences_found:
        if len(seq) == 0:
            continue
        seq_bboxes = [data_item['bboxes'][x] for x in seq]
        seq_bbox   = combine_bboxes(seq_bboxes)
        entity_boxes.append(seq_bbox)
    return entity_boxes

def add_tight_boxes_to_imgj_data_item(data_item):
    data_item['entity_boxes'] = get_tight_boxes_to_imgj_data_item(data_item)


def add_tight_boxes(img_json_data):
    for data_item in img_json_data:
        add_tight_boxes_to_imgj_data_item(data_item)



Bbox = Tuple[Number,Number,Number,Number]
def normalize_bbox(bbox:Bbox, width:Number, height:Number) -> Bbox:
    return tuple([
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ])

def normalize_bboxes(bboxes:Sequence[Bbox], width:Number, height:Number) -> List[Bbox]:
    return [normalize_bbox(bbox, width, height) for bbox in bboxes]

def annotation_block_label_to_adj_mat(annotation_block_labels):
    a = np.array(annotation_block_labels)
    adj_mat = a[None,:] == a[:,None]
    adj_mat[a == -1,:] &= False
    adj_mat[:,a == -1] &= False
    np.fill_diagonal(adj_mat, False)
    return adj_mat.astype(int)

def adj_mat_to_adj_mat(adjmatA, adjmatAtoB, do_not_integerize=False):
    A = adjmatA # shape nxn
    AtoB = adjmatAtoB # shape nxm. AtoB[i,j] == 1 if ith from A is equivalent to jth from B
    R = AtoB.T.dot(A).dot(AtoB)
    if not do_not_integerize:
        R = (R > 0).astype(int)
    return R

def wordToTokenList_to_wordToTokenAdjMat(wordToTokenLists,nWords=None):
    # It is necessary to take nWords as arguments because sometimes, a word might not get mapped to a token
    word_to_token_list = [*it.chain.from_iterable(wordToTokenLists)]
    wtklistwithout_1 = [x for x in word_to_token_list if x >= 0]
    if nWords is None:
        nWords = max(wtklistwithout_1) - min(wtklistwithout_1) + 1
    nTokens = len(word_to_token_list)
    m = np.zeros((nWords,nTokens))
    token_idx_word_idx = np.array([*enumerate(word_to_token_list)])
    token_idx_word_idx = token_idx_word_idx[token_idx_word_idx[:,1] != -1]
    token_idx, word_idx = token_idx_word_idx.T
    m[word_idx, token_idx] = 1
    return m

ImageJsonDataPoint = Dict[str,Any]

def windowing_indices(original_length :int, window_overlap:int, window_size:int) -> List[Tuple[int,int]]:
    window_stride = window_size - window_overlap
    starting_points = range(0,original_length, window_stride)
    ending_points = [x + window_size for x in starting_points]
    intermediate_pairs = [(t1,t2) for (t1,t2) in zip(starting_points,ending_points) if t2 <= original_length]
    if intermediate_pairs[-1][1] < original_length:
        t2 = original_length
        t1 = original_length - window_size
        intermediate_pairs.append((t1,t2))
    return intermediate_pairs

