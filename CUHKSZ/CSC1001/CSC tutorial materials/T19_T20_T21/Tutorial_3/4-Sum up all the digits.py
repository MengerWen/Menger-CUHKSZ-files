##Prompt up the indicator for user to enter
number=eval(input("Enter a number between 0 and 1000:"))

# ##Use the variable to do the operation
# sumup=0
# sumup+=number%10
# number//=10
# sumup+=number%10
# number//=10
# sumup+=number%10
#
# ##Display the result
# print("Sum up all the digits as:",sumup)






#
# another solution?
number = input("Enter a number between 0 and 1000:")
str_len = len(number)
sumup = 0
for i in range(str_len):
    sumup += eval(number[i])
print(sumup)