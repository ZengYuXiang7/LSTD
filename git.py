# coding : utf-8
# Author : yuxiang Zeng
import pexpect
import subprocess
import sys
import os
import pickle

def git_first_push(url):
    subprocess.run(f"git branch -M main", shell=True)
    subprocess.run(f"git commit -am 'First Commit'", shell=True)
    subprocess.run(f"git remote add {url}", shell=True)
    subprocess.run(f"git config --global credential.helper store", shell=True)
    subprocess.run(f"git push -u origin main", shell=True)

def git_push(message):
    subprocess.run(f'git commit -am "{message}"', shell=True)
    subprocess.run(f"git push", shell=True)

def git_pull():
    subprocess.run(f'git pull', shell=True)

def git_update(message='Commit the work before the pull'):
    subprocess.run(f'git commit -am {message}', shell=True)
    subprocess.run(f'git pull', shell=True)

def git_reset(cnt):
    subprocess.run(f"git reset main{'^' * cnt}", shell=True)

if __name__ == "__main__":
    inputs = input('push or pull or reset or update? : ').strip()
    if inputs == 'pull':
        git_pull()
    elif inputs == 'push':
        message = input('message : ').strip()
        git_push(message)
    elif inputs == 'reset':
        cnt = int(input('number of commits : ').strip())
        git_reset(cnt)
    elif inputs == 'update':
        message = input('message (if no, just enter!): ').strip()
        git_update(message)

