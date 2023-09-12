FROM python:3.10

WORKDIR /code

RUN pip install --upgrade pip
RUN pip install --upgrade build

COPY . .
RUN python3 -m build
RUN pip install .

CMD ["/bin/bash"]
