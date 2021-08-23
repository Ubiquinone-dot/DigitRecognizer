from src.Network1 import NET

def main():
    nn = NET()
    nn.load_data()
    nn.train()


if __name__ == '__main__':
    main()