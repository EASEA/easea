# colors
_GREEN=$'\x1b[32m'
_END=$'\x1b[0m'

# get Os name
UNAME:=$(shell uname)

install:
	@echo "\n##### Installing the modules #####\n"

	npm install

	@echo "\n##### Beginning of the packaging #####\n"	

	@echo "\n##### Initialization #####\n"

	rm -rf deploy dist

	npx nodegui-packer --init EASEA-compiler-app

	@echo "\n##### Adding additional files #####\n"

ifeq ($(UNAME), Linux)
	cp -rf src/plot.py src/3D_plot.py documentation/ deploy/linux/EASEA-compiler-app/
	cp -Rf src/ deploy/linux/EASEA-compiler-app/
endif

ifeq ($(UNAME), Darwin)
	cp -rf src/plot.py src/3D_plot.py documentation/ deploy/darwin/EASEA-compiler-app/
	cp -Rf src/ deploy/darwin/EASEA-compiler-app/
endif

	@echo "\n##### Build dist #####\n"

	npm run build

	@echo "\n##### Packaging in progress #####\n"

	npx nodegui-packer --pack ./dist
 
	find 'deploy/' -type d -name doc | sed 's/doc//' | xargs -n 1 cp -rt ./
	rm -rf deploy/ dist/

	mv EASEA-compiler-app/*.AppImage EASEA-compiler-app/easea-compiler-app
	rm -f EASEA-compiler-app/documentation/interactive*

	@echo "\n##### Packaging finished #####\n"

	@echo "\n##### Installing executable #####\n"

	sudo cp EASEA-compiler-app/easea-compiler-app /usr/local/easena/bin/easea-compiler-app

	@printf "\033[32m\n##### Installation complete #####\n\n\033[0m"

update-documentation:
	asciidoctor documentation/*.adoc

uninstall:
	rm -rf deploy dist EASEA-compiler-app

	@echo "\n##### Uninstalling executable #####\n"

	sudo rm -rf /usr/local/easena/bin/easea-compiler-app

	@printf "\033[32m\n##### Uninstallation completed #####\n\n\033[0m"
