
13 lines (11 sloc)  222 Bytes
 
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"prernapandey@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml