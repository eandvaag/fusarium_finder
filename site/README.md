# Fusarium Finder Node.js Application


## Install (Non-Docker)


Install Node.js.
```
curl -fsSL https://deb.nodesource.com/setup_14.x | sudo -E bash - &&\
sudo apt-get install -y nodejs
sudo npm install -g n
sudo n 14.18.1
```

Install packages (execute from `site/myapp/` directory).
```
npm install package.json
```

Install PostGreSQL.
```
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update
sudo apt-get -y install postgresql
```

Acquire config.json and place in `site/myapp/config`.

Create database and database user.
```
sudo -u postgres psql
```
In psql (replace 'XXX' with password found in `config.js`):
```
CREATE DATABASE fusariumfinder_db;
CREATE ROLE fusariumfinder_db_user WITH PASSWORD 'XXX';
ALTER ROLE "fusariumfinder_db_user" WITH LOGIN;
```

Grant privileges to fusariumfinder_db_user.
```
postgres=# \c fusariumfinder_db 
fusariumfinder_db=# GRANT ALL ON SCHEMA public TO fusariumfinder_db_user;
```


Run database migrations from the `site/myapp` directory:
```
npx sequelize-cli db:migrate
```

Acquire user seeders file and place in `site/myapp/seeders`. Then run seeders.
```
npx sequelize-cli db:seed:all
```


Add environment variables to `~/.bashrc`.
```
export FF_IP="YOUR_IP_ADDRESS_HERE"
export FF_PORT="8130"
export FF_PY_PORT="8131"
export FF_PATH="/fusariumfinder"
export FF_API_KEY="YOUR_SECRET_API_KEY_HERE"
```


Acquire cert.pem and key.pem and add to `site/myapp` directory.


Install ImageMagick.
```
sudo apt install imagemagick
```

Edit /etc/ImageMagick-6/policy.xml to allow larger files to be converted.
```
<policy domain="resource" name="disk" value="10GiB"/>
```



Install other apt packages.
```
sudo apt install libimage-exiftool-perl
sudo apt install libvips-tools
sudo apt install libgdal-dev gdal-bin
```


Create symlink for storing user data.
```
ln -s backend/src/usr site/myapp/usr
```


To start the Node.js application, execute the following command from the `site/myapp` directory:
```
DEBUG=myapp:* npm start
```