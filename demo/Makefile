include ./include/Makefile.inc

SOURCE_CU   = $(shell find . -name '*.cu' 2>/dev/null)
SOURCE_PY   = $(shell find . -name '*.py' 2>/dev/null)
OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)
TARGET_SO   = $(SOURCE_CU:.cu=.so)

-include $(DEP)
INCLUDE += -Iinclude
INCLUDE += -I/usr/local/cuda-12.2/targets/x86_64-linux/include
INCLUDE += -I/usr/local/tensorrt/include

all: $(TARGET_SO)

%.so: %.o
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $+

%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

.PHONY: test
test:
	make clean
	make
	python3 $(SOURCE_PY)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.exe ./*.plan