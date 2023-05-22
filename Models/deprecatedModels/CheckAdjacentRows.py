import csv

def compare_labels(csv_file_path, n, previousOnly=False, countExtraRows=False):
    count = 0
    
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader]
        
        for i in range(len(rows)):
            current_row = rows[i]
            
            true_label = int(float(current_row['True_label']))
            predicted_label = int(current_row['Predicted_label'] == 'True') # This will be 1 if True, 0 if False
            
            if true_label == 1:
                countCheck = count
                if true_label == predicted_label:
                    count += 1
                    #print(i+2)
                if countCheck == count or countExtraRows:
                    for j in range(1, n+1):
                        if (i+j < len(rows) and not previousOnly) and (countCheck == count or countExtraRows):
                            adjacent_row = rows[i+j]
                            adjacent_predicted_label = int(adjacent_row['Predicted_label'] == 'True')
                            if true_label == adjacent_predicted_label:
                                count += 1
                                #print(i+2)
                        if i-j >= 0 and (countCheck == count or countExtraRows):
                            adjacent_row = rows[i-j]
                            adjacent_predicted_label = int(adjacent_row['Predicted_label'] == 'True')
                            if true_label == adjacent_predicted_label:
                                count += 1
                                #print(i+2)
                
    return count


def main():
    csv_file_path = 'RELEARNBackEnd\Processing\Models\predictions_un_tr_1236_t_5.csv'
    n = 0  # Number of adjacent rows to look at
    count = compare_labels(csv_file_path, n)
    print(f'Total TP within 50 ms: {count}')


if __name__ == '__main__':
    main()
