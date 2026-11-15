# Deploying the dashboard

The dashboard is a Flask (Dash) application. As in the paper (Section 7.1) it
is served by Gunicorn behind Nginx on an Ubuntu machine. It only needs the six
`app_*.csv` tables; the rest of the pipeline can run elsewhere.

Memory: about 300 MB per worker after start-up with the 2020 data (measured on
Windows; allow more while requests are served). Disk: about 45 MB for the data.

## 1. Install

```bash
sudo adduser --system --group --home /opt/stroo stroo
sudo -u stroo git clone https://github.com/nomoteticus/reddit-disinformation.git /opt/stroo
cd /opt/stroo
sudo -u stroo python3 -m venv venv
sudo -u stroo venv/bin/pip install -e ".[dashboard]"
```

The package must be installed in editable mode (`-e`) from the cloned
repository, because Nginx serves the stylesheets from `src/stroo/dashboard/assets`.

## 2. Data

Put the six tables in `/var/lib/stroo/data/dashboard/`:

```bash
sudo mkdir -p /var/lib/stroo/data/dashboard
# copy app_flags_day.csv, app_subm_day.csv, app_subr_week.csv,
#      app_authFlaggers_week.csv, app_authFlagged_week.csv, app_domain_week.csv
sudo chown -R stroo /var/lib/stroo
```

The 2020 tables used in the paper are on Figshare:
https://doi.org/10.6084/m9.figshare.13174136

## 3. Check it starts

```bash
sudo -u stroo STROO_DATA_DIR=/var/lib/stroo/data venv/bin/gunicorn --bind 127.0.0.1:8000 stroo.dashboard.wsgi:server
curl -I http://127.0.0.1:8000/        # expect HTTP 200
```

## 4. Run as a service behind Nginx

```bash
sudo cp deploy/stroo-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now stroo-dashboard

sudo apt install nginx
sudo cp deploy/nginx-stroo.conf /etc/nginx/sites-available/stroo
sudo ln -s /etc/nginx/sites-available/stroo /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

For HTTPS, add a certificate with `certbot --nginx` once a host name points at
the machine.

## 5. Updating the data

The app reads the tables once, at start-up. After replacing them:

```bash
sudo systemctl restart stroo-dashboard
```

`crontab.example` shows the daily schedule used while the dashboard was live.

## On Windows (local testing only)

Gunicorn does not run on Windows. Use the built-in server (`stroo dashboard`)
or Waitress:

```bash
waitress-serve --port=8050 stroo.dashboard.wsgi:server
```

## What has been tested

The WSGI entry point `stroo.dashboard.wsgi:server` was run under Waitress on
Windows with the 2020 data and answered requests. The systemd unit and the
Nginx configuration follow the standard Gunicorn + Nginx setup but have not
been run on a server as part of this clean-up.
