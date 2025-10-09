def my_split(sentence, seperator):
    final_result=[]
    temp=""
    for char in sentence:
        if char==seperator:
            if temp!="":
                final_result.append(temp)
                temp=""
        else:
            temp+=char
    if temp!="":
        final_result.append(temp)
    return final_result

def my_join(list, seperator):
    final_result = ""
    for i in range(len(list)):
        final_result += list[i]
        if i != len(list) - 1:
            final_result += seperator
    return final_result


sentence = input("Please enter sentence: ")
words = my_split(sentence, " ")
joined = my_join(words, ",")
print(joined)

for w in words:
    print(w)