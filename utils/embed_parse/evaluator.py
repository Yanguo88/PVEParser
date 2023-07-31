from scipy.special import perm,comb
import numpy as np
from sklearn.metrics import accuracy_score
from nltk.metrics.distance import edit_distance
import scipy
def get_metrics(groundtruth, parsed_ids, debug=False):
    """ Compute accuracy metrics between log parsing results and ground truth

    Arguments
    ---------
        groundtruth : pandas.Series
            A sequence of groundtruth event Ids
        parsed_ids : pandas.Series
            A sequence of parsed event Ids
        debug : bool, default False
            print error log messages when set to True

    Returns
    -------
        precision : float
        recall : float
        f_measure : float
        accuracy : float
    """
    # total = tp+fp+fn+tn
    total = comb(groundtruth.size,2)
    groundtruth_valuecounts = groundtruth.value_counts()
    # tp_plus_fn
    tp_plus_fn = 1
    for count in groundtruth_valuecounts:
        if count > 1:
            tp_plus_fn += comb(count, 2)
    parsedlog_valuecounts = parsed_ids.value_counts()
    # tp_plus_fp
    tp_plus_fp = 1
    for count in parsedlog_valuecounts:
        if count > 1:
            tp_plus_fp += comb(count, 2)
    #tp
    tp = 0
    accurate_events = 0 # determine how many lines are correctly parsed
    for parsed_eventId in parsedlog_valuecounts.index:
        #解析日志集中，一个eventId 对应的下标
        logIds = parsed_ids[parsed_ids == parsed_eventId].index
        #这些下标在groundtruth中对应的eventId，并进行统计
        groundtruth_logId_valuecounts = groundtruth[logIds].value_counts()
        #
        error_eventIds = (parsed_eventId, groundtruth_logId_valuecounts.index.tolist())
        error = True
        # 如果在groundtruth 中这些下标对应的日志为同一eventId,则表示解析准确
        if groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = groundtruth_logId_valuecounts.index[0]
            if logIds.size == groundtruth[groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False
        #打印解析错误的相关信息学
        if error and debug:
            print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')
        # tp:同一类日志被分到同一簇下，不管是否解析正确
        for count in groundtruth_logId_valuecounts:
            if count > 1:
                tp += comb(count, 2)
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = total - fp - tp - fn

    precision = float(tp) / tp_plus_fp
    recall = float(tp) / tp_plus_fn
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / groundtruth.size
    rand_index = float(tp+tn)/total
    return precision, recall, f_measure, accuracy,rand_index


def evaluate(groundtruth, parsedresult,islower =True):
    """ Evaluation function to benchmark log parsing accuracy

    Arguments
    ---------
        groundtruth : str
            file path of groundtruth structured csv file
        parsedresult : str
            file path of parsed structured csv file

    Returns
    -------
        f_measure : float
        accuracy : float
    """
    # df_groundtruth = pd.read_csv(groundtruth)
    # df_parsedlog = pd.read_csv(parsedresult, index_col=False)
    #
    df_groundtruth = groundtruth
    df_parsedlog = parsedresult

    # Remove invalid groundtruth event Ids
    null_logids = df_groundtruth[~df_groundtruth['EventId'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]
    if islower:
        df_groundtruth.EventTemplate = df_groundtruth.EventTemplate.str.lower()
    accuracy_exact_string_matching = accuracy_score(np.array(df_groundtruth.EventTemplate.values, dtype='str'),
                                                    np.array(df_parsedlog.EventTemplate.values, dtype='str'))

    edit_distance_result = []
    for i, j in zip(np.array(df_groundtruth.EventTemplate.values, dtype='str'),
                    np.array(df_parsedlog.EventTemplate.values, dtype='str')):
        edit_distance_result.append(edit_distance(i, j))

    edit_distance_result_mean = np.mean(edit_distance_result)
    edit_distance_result_std = np.std(edit_distance_result)

    (precision, recall, f_measure, accuracy_PA) = get_accuracy(df_groundtruth['EventId'], df_parsedlog['EventId'])
    print(
         'accuracy_exact_string_matching: %.4f, edit_distance_result_mean: %.4f, edit_distance_result_std: %.4f' % (
          accuracy_exact_string_matching, edit_distance_result_mean,
        edit_distance_result_std))

    return accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean, edit_distance_result_std


def rand_index_score(clusters, classes):

    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    """ Compute accuracy metrics between log parsing results and ground truth

    Arguments
    ---------
        series_groundtruth : pandas.Series
            A sequence of groundtruth event Ids
        series_parsedlog : pandas.Series
            A sequence of parsed event Ids
        debug : bool, default False
            print error log messages when set to True

    Returns
    -------
        precision : float
        recall : float
        f_measure : float
        accuracy : float
    """
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += scipy.special.comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += scipy.special.comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False
        if error and debug:
            print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += scipy.special.comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size

    return precision, recall, f_measure, accuracy