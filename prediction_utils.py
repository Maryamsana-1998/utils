import itertools
from pathlib import Path

import fitz
import numpy as np
import regex as re
import scipy.special
import torch
from scipy.sparse.csgraph import connected_components

from cache_module import mem
from loguru_logger import logger
from util import combine_bboxes, get_pdf_data


@mem.cache
def pdf_to_image_jsons(result_folder, pdf_path):
    results = []
    doc = fitz.open(str(pdf_path))
    for idx in range(len(doc)):
        ocr_data, image = get_pdf_data(pdf_path, idx)
        # print(ocr_data)
        ann_data= {
            'annotation_bboxes' : [],
            'annotation_labels' : [],
            'annotation_size'   : ocr_data['size'],
            'bilou_labels': ['O' for x in ocr_data['words']],
            'annotation_block_label':[0 for x in ocr_data['words']]
        }
        fin_data = {}
        fin_data.update(ocr_data)
        fin_data.update(ann_data)
        logger.debug(fin_data.keys())
        imgpath = Path(result_folder)/ (Path(pdf_path).stem + f'_{str(idx).zfill(3)}.png')
        fin_data['image_path'] = str(imgpath)
        image = image.resize(fin_data['size'])
        image.save(imgpath)
        results.append(fin_data)
    return results

def pdfs_to_image_jsons(result_folder, pdf_paths):
    results = []
    for pdf_path in pdf_paths:
        results.extend( pdf_to_image_jsons(result_folder, pdf_path) )
    return results


# for j, test_idx in enumerate(all_idxs):
def batched_model_prediction_over_image_json(class_names, image_json, model, batch_size=8,stride=480):
    from dataloader import Indexable_LayoutLMv2_Dataset, LayoutLMv2_Dataset
    encoding_dataset = Indexable_LayoutLMv2_Dataset(
        LayoutLMv2_Dataset([image_json], chosen_classes=class_names,do_not_truncate=True,stride=stride),
        batch_size=batch_size,shuffle=False)
    logger.debug(f"dataset has len: {len(encoding_dataset)}")
    logger.debug(f"json path= {image_json['image_path']}")
    # encoding = dataset[j]
    # original_data = dataset.json_data[j]
    original_data = image_json
    with torch.no_grad():
        prediction_list = []
        word_mapping_batch_list = []
        for encoding in encoding_dataset:
            relevant_prediction = model(**encoding).logits.detach().cpu().numpy()
            logger.debug(f"relevant_prediction.shape:{relevant_prediction.shape}")
            prediction_list.append(relevant_prediction)
            word_mapping_batch_list .append( encoding['word_mapping'].detach().cpu().numpy() )

        prediction_list_shapes = [predictionx.shape for predictionx in prediction_list]
        logger.debug(f"Shapes of prediction list {prediction_list_shapes}")
        prediction = np.concatenate(prediction_list,axis=0)
        logger.debug(f"Shape of prediction {prediction.shape}")
        assert prediction.shape[0] == sum([x[0] for x in prediction_list_shapes])

        word_mapping_batch_list_shapes = [word_mapping_batch_list_x.shape for word_mapping_batch_list_x in word_mapping_batch_list]
        logger.debug(f"Shape of word_mapping {word_mapping_batch_list_shapes}")
        word_mapping = np.concatenate(word_mapping_batch_list,axis=0)
        logger.debug(f"Shape of word_mapping {word_mapping.shape}")
        assert word_mapping.shape[0] == sum([x[0] for x in word_mapping_batch_list_shapes])
    return prediction ,word_mapping

def word_mapping_to_uniq_word_idxs(word_mapping):
    flattened_word_mapping = word_mapping.flatten()
    flattened_word_mapping = flattened_word_mapping[flattened_word_mapping != -1]
    return sorted(list(set(flattened_word_mapping.tolist())))

def word_predictions_from_token_predictions(relevant_prediction, word_mapping, n):
    pred_word_labels = []
    for i in range(n):
        logger.debug(f"i is currently {i}")
        batch_span, token_span = (word_mapping == i).nonzero()
        logger.debug(f"batch_span:{batch_span}")
        logger.debug(f"token_span:{token_span}")
        # print(f"{i}->{token_span}")
        #TODO: resolve none token bug in training and inference
        word_preds = relevant_prediction[batch_span,token_span,:]
        logger.debug(f"word_preds.shape:{word_preds.shape}")
        try:
            word_preds = np.max(word_preds, axis=0)
        except ValueError as e:
            logger.warning(e)
            logger.debug(f"Continuing at word idx {i} because of valueerror")
            pred_word_labels.append(None)
            continue
        word_preds = scipy.special.softmax(word_preds)
        max_pred_idx = np.argmax(word_preds)
        confidence = word_preds[max_pred_idx]
        pred_word_labels.append(max_pred_idx)

    return pred_word_labels

def bilou_predictions_to_sequences(pred_word_labels):
    
    sequences_found = []
    seqstr = ''.join([x[0] for x in pred_word_labels])
    complete_seq_re_BIL = re.compile(r'BI*L')
    matches = [[*range(*x.span())] for x in complete_seq_re_BIL.finditer(seqstr, overlapped=True)]
    complete_seq_re_XIL = re.compile(r'[^I](I*L)')
    # print([x for x in complete_seq_re_XIL.finditer(seqstr, overlapped=True)])
    matches += [[*range(x.span()[0]+1,x.span()[1])] for x in complete_seq_re_XIL.finditer(seqstr, overlapped=True)]
    complete_seq_re_BIX = re.compile(r'BI*([^I])')
    matches += [[*range(x.span()[0],x.span()[1]-1)] for x in complete_seq_re_BIX.finditer(seqstr, overlapped=True)]
    # print((matches))
    n  = len(pred_word_labels)
    adj = np.zeros((n,n))
    for seq in matches:
        for i, j in itertools.product(seq,seq):
            adj[i,j] = 1
            adj[j,i] = 1
        
    assert (adj == adj.T).all()
    
    n_coms, wlbs = connected_components(adj)
    assert len(set(wlbs)) == n_coms
    # print([*enumerate(wlbs)])
    sequences_found = []
    for uwlb in set(wlbs):
        sequences_found.append([i for i,x in enumerate(wlbs) if (x == uwlb) and seqstr[i] != 'O'])
    return sequences_found

def bilou_sequences_to_entity_boxes(bboxes, sequences_found):
    entity_boxes = []
    for seq in sequences_found:
        if len(seq) == 0:
            continue
        seq_bboxes = [bboxes[x] for x in seq]
        seq_bbox   = combine_bboxes(seq_bboxes)
        entity_boxes.append(seq_bbox)
    return entity_boxes    

def bilou_predictions_to_entity_boxes(bboxes, pred_word_labels):
    assert len(bboxes) == len(pred_word_labels)
    sequences_found = bilou_predictions_to_sequences(pred_word_labels)
    return bilou_sequences_to_entity_boxes(bboxes, sequences_found)


# for j, test_idx in enumerate(all_idxs):
def model_predictions_over_imagejson(imagejson ,class_names, model, id2label ):
    relevant_prediction, word_mapping = batched_model_prediction_over_image_json(class_names, imagejson, model)
    imagejson['relevant_prediction'] = relevant_prediction
    imagejson['word_mapping'] = word_mapping
    # word_mapping = encoding['word_mapping'].detach().cpu().numpy()
    pred_word_idxs = word_predictions_from_token_predictions(relevant_prediction, word_mapping, len(imagejson['words']))
    pred_word_labels = [id2label.get(max_pred_idx,'O') for max_pred_idx in pred_word_idxs]

    imagejson['word_preds'] = pred_word_labels
    block_no = imagejson['block_no']
    sequences_found = []
    
            # blended = Image.blend(image, overlay, 0.8) 
    block_no = imagejson['block_no']
    imagejson['word_preds'] = pred_word_labels
    entity_boxes = bilou_predictions_to_entity_boxes(imagejson['bboxes'], pred_word_labels)
    imagejson['entity_boxes'] = entity_boxes
