CC = gcc

APP_PATH = .

LIB_PATH = /usr/lib/x86_64-linux-gnu
LIB_NAME = libOpenCL.so

INC_DIR  = $(APP_PATH)/inc
BIN_DIR  = $(APP_PATH)/bin
MMIO_DIR = $(APP_PATH)/mmio
OBJ_DIR  = $(APP_PATH)/obj

TARGETS = coo csr ell sigma_c
HEADERS = $(INC_DIR)/helper_functions.h

INCLUDES = -I$(MMIO_DIR) -I$(INC_DIR)
LDFLAGS  = -L$(LIB_PATH) -l:$(LIB_NAME)
CFLAGS   = -Wall -Werror

default: all

all: $(TARGETS)

mmio.o: $(MMIO_DIR)/mmio.c
	@mkdir -p $(OBJ_DIR)
	$(CC) -c $< $(CFLAGS) -o $(OBJ_DIR)/$@

%.o: %.c
	@mkdir -p $(OBJ_DIR)
	$(CC) -c $< $(CFLAGS) $(INCLUDES) -o $(OBJ_DIR)/$@
	
$(TARGETS): % : %.o mmio.o $(HEADERS)
	@mkdir -p $(BIN_DIR)
	@echo "Compile $@"
	$(CC) $(OBJ_DIR)/$< $(OBJ_DIR)/$(word 2,$^) $(CFLAGS) $(LDFLAGS) -o $(BIN_DIR)/$@
	@echo "$@ compiled"
	
clean:
	$(RM) -r $(OBJ_DIR)
	$(RM) -r $(BIN_DIR)
