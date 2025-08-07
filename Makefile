### pipeline ###
install:
	composer install

generate:
	./vendor/bin/daux generate

serve:
	./vendor/bin/daux serve
