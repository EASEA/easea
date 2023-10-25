
# colors
_GREEN=$'\x1b[32m'
_END=$'\x1b[0m'

# get Os name
UNAME:=$(shell uname)

install:

# test LOC existence
ifdef LOC
	@if [ -d $(LOC) ]; then printf "$(LOC) is valid\n"; else printf "$(LOC) does not exists\n"; exit 1; fi
endif

# remove last versions
	@echo "\n##### Installing modules #####\n"

	npm install

	@echo "\n##### Installing python dependencies #####\n"

	@echo "\n##### Beginning of the packaging #####\n"	

	@echo "\n##### Initialization #####\n"

	@rm -rf deploy dist

	npx nodegui-packer --init guide

	@echo "\n##### Adding additional files #####\n"

ifeq ($(UNAME), Linux)
	@cp -rf src/plot.py src/3D_plot.py documentation/ deploy/linux/guide/
	@cp -Rf src/ deploy/linux/guide/
endif

ifeq ($(UNAME), Darwin)
	@cp -rf src/plot.py src/3D_plot.py documentation/ deploy/darwin/guide/
	@cp -Rf src/ deploy/darwin/guide/
endif

	@echo "\n##### Build dist #####\n"

# adapt memory to build with python packages
# export NODE_OPTIONS=--max_old_space_size=32768

	npm run build

	@echo "\n##### Packaging in progress #####\n"

	npx nodegui-packer --pack ./dist
 
	@find 'deploy/' -type d -name doc | sed 's/doc//' | xargs -n 1 cp -rt ./
	@rm -rf deploy/ dist/

	@mv guide/*.AppImage guide/guide
	@rm -f guide/documentation/interactive*

	@echo "\n##### Packaging finished #####\n"

	@echo "\n##### Installing executable #####\n"

	@if test -f /usr/local/easea/bin/guide; then sudo rm -f /usr/local/easea/bin/guide; fi
	@if test -f /usr/local/easea/bin/easea-compiler-app; then sudo rm -f /usr/local/easea/bin/easea-compiler-app; fi

ifdef LOC
	@sudo cp guide/guide $(LOC)
else
	@sudo cp guide/guide /usr/local/easea/bin/guide
endif

	@printf "\033[32m\n##### Installation complete #####\n\n\033[0m"

update-documentation:
	asciidoctor documentation/*.adoc

uninstall:
	@rm -rf deploy dist guide

	@echo "\n##### Uninstalling executable #####\n"

	@sudo rm -rf /usr/local/easea/bin/guide

	@printf "\033[32m\n##### Uninstallation completed #####\n\n\033[0m"

clean:
	rm -rf src/python_modules

