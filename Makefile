### pipeline ###
install:
	composer install

generate:
	./vendor/bin/daux generate

serve:
	./vendor/bin/daux serve

clean:
	rm -rf static vendor composer.lock