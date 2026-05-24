import pandas as pd

"""    
This code is used to compute the Average Precision (AP) by sorting based on the post-integration confidence scores.

Dataset Chinese Explanations:
    'OfficialReception(公务接待)', ' OfficialVehicles(公务用车)' '  ConventionExpense(会议培训)',
    ' Remuneration(人员薪酬)'' PropertyManagement(物业管理)'' Procurement(设备/服务采购)',' NULL(空)',
    are the class label in the dataset.
    'Purpose(用途)', 'Category(类别)' are the column headers in the table.
"""



def main():
    df_data = pd.read_excel('../../result/result/finetune_one_class_score_yixing.xlsx')
    class_label = ['公务接待', '公务用车', '会议培训', '人员薪酬', '物业管理', '设备/服务采购']
    results = []

    ap1 = apk_single_class(df_data, class_label, k_max=500)
    ap2 = apk_single_class(df_data, class_label, k_max=1000)
    ap3 = apk_single_class(df_data, class_label, k_max=2000)
    #ap3 = apk_single_class(df_data, class_label, k_max=len(df_data))
    results.append({
            'ap@100': ap1,
            'ap@200': ap2,
            'ap@all': ap3
        })
    for result in results:
        print(f"ap@100: {result['ap@100']:.3f}, ap@200: {result['ap@200']:.3f}, ap@all: {result['ap@all']:.3f}")


def apk_single_class(df_data, class_label, k_max=0):

    df_sorted = df_data.sort_values(by='confidence', ascending=False)
    y_pred = df_sorted['类别'].values.tolist()
    if k_max != 0:
        y_pred = y_pred[:k_max]
    y_true = [x for x in y_pred if x in class_label]


    correct_predictions = 0
    running_sum = 0

    # Count how many target_class are in y_true
    # num_relevant = y_true.count(target_class)
    num_relevant = len(y_true)

    if num_relevant == 0:
        return 0.0

    for i, yp_item in enumerate(y_pred):
        k = i + 1  # Position starts from 1

        if yp_item in y_true:
            correct_predictions += 1
            running_sum += correct_predictions / k

    return running_sum / num_relevant


if __name__=='__main__':
    main()