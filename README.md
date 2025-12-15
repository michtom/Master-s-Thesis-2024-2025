# Utilization of Machine Learning Methods in the Cryptocurrency with Consideration of Sentiment Models
### Master's Thesios source code
#### Faculty of Mathematics and Information Science, Warsaw University of Technology
##### December 2025
##### Authors: Michał Tomczyk, with the supervision of dr Robert Małysz

This repository contains all necessary source code and analysis notebooks created for the purpose of the thesis.

#### How to set up locally
* the code was written uising Python 3.10. Install this version of the programming language
* clone this repository
* install the requirements.txt file: `pip install -r requirements.txt` (when being in the repository's root directory locally)
* extend the PYTHONPATH environmental variable to contain the repository's root: on Linux and MacOS: `export PYTHONPATH="{path_to_repository_root}:$PYTHONPATH"`, on Windows: `$env:PYTHONPATH="{path_to_repository_root};$env:PYTHONPATH"`
* create `credentials.py` file in `config` folder. Set the values of `reddit_client_id`, `reddit_agent` `reddit_secret` and `api_key` (as plain python strings). Either contact the author for credentials or [set up OAuth2 for Reddit](https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps) (for the first three variables) and [generate API key from Finlight.me](https://finlight.me) (the fourth variable)
* either collect your own data using scripts in `data_retrieving` folder or contact the author for dataset
* please note that most notebooks were used via Google Colab platform and were written to be used in this envronment. Should you try to executre them, ether delete cells used for mounting Google Drive or copy the root directory's content to a directory called `magisterka` in your home directory in Google Drive
