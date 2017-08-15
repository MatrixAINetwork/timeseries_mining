def table2markdown(file_name, table_list, description=None, mode='w'):
    """
    
    :param file_name: 
    :param table_list: 
    :param description: 
    :param mode: 
    :return: 
    """
    print("out put to file %s ...." % file_name)
    fout = open(file_name, mode)
    if description is not None:
        fout.write(description)

    str_head = ""
    str_gap = ""
    for item in table_list[0]:
        str_head += item + ' | '
        str_gap += '---|'
    fout.write(str_head)
    fout.write("\n")
    fout.write(str_gap)
    fout.write("\n")

    for i in range(1, len(table_list)):
        item = table_list[i]
        str_line = ""
        for ii in item:
            str_line += str(ii) + ' | '
        fout.write(str_line)
        fout.write("\n")

    fout.write("\n")


def headmarkdown(head):
    str_head = ""
    str_gap = ""
    for item in head:
        str_head += item + ' | '
        str_gap += '---|'
    return str_head + '\n' + str_gap + '\n'


def row2markdown(row):
    return row2str(row, ' | ')


def row2str(row, delimiter=','):
    str_line = ""
    for item in row:
        str_line += str(item) + delimiter
    return str_line[:len(str_line)-len(delimiter)]


def data2file(filename, data, delimiter=',', mode='w'):
    file = open(filename, mode)
    for i in range(len(data)):
        r_str = row2str(data[i], delimiter)
        file.write(r_str+'\n')


