import csv

solution_path = "./submission_fusion.csv"
standart_list = "../data/datalists/testlist.txt"

predicted = {}

with open(solution_path, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    headers = next(csv_reader)
    for row in csv_reader:
        predicted[row[0]] = row[1]

    # print(f'Processed {line_count} lines.')

result = []
with open(standart_list, "r") as standart_list:
    for line in standart_list:
        try:
            if line.strip().split("/")[0] == "HandstandPushups":
                pred_true_name = line.strip().replace("HandStandPushups", "HandstandPushups")
                result.append((line.strip(), predicted[pred_true_name]))
            else:
                result.append((line.strip(),predicted[line.strip()]))
        except:
            print(line)

with open("./new_submission_fusion.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['video', 'label'])
    writer.writerows(result)



