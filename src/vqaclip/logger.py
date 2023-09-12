class IOStream():
    def __init__(self, path: str) -> None:
        self.f = open(path, 'a')

    def cprint(self, text: str) -> None:
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self) -> None:
        self.f.close()