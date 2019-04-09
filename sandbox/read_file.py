cla = eval(open('notmnist_cla_unc.txt',mode='r').read())
test_ll = eval(open('notmnist_testll.txt',mode='r').read())

for i in range(10):
	print("test_ll for task 0 is ", test_ll[i])
	print("cla_unc for task 0 is ", cla[i])
