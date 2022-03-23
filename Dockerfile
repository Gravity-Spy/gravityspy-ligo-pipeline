FROM ubuntu:groovy-20210225 AS spython-base
RUN apt-get -y update && apt-get -y install python3-pip git
RUN https://github.com/Gravity-Spy/gravityspy-ligo-pipeline /gravityspy-ligo-pipeline
RUN pip3 install /gravityspy-ligo-pipeline
CMD python3 -c "from gravityspy_ligo.mymodule.mymodule import MyClass;print(MyClass().return_one())"
