class TEST:
    def __init__(self, test) -> None:
        self.test = test

    def sum(self):
        return self.test + self.test  

    def dif(self):
        return self.test -self.test 

    def div(self):
        return self.test /self.test 

    def mul(self):
        return self.test *self.test 

    def pwr(self, amt):
        print("HEJ", amt)
        return self.test**amt

def main():
    obj = TEST(1)
    testlis = [obj.sum(), obj.sum(), obj.div(), obj.sum(), obj.sum(), obj.mul(), obj.pwr(4), obj.dif()]
    for i in testlis:
        print(i)

main()
