import os

def main():
    needed = []
    with open('needed.txt', 'r') as file:
        for line in file.readlines():
            needed.append(line.replace('\n', ''))

    needed = set(needed)

    for file in os.listdir('./data/images'):
        if file not in needed:
            os.remove(os.path.join('./data/images', file))


if __name__ == '__main__':
    main()