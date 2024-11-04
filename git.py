# coding : utf-8
# Author : yuxiang Zeng
import pexpect
import subprocess
import sys
import os
import pickle

def git_first_push():
    subprocess.run(f"git add --all", shell=True)
    subprocess.run(f"git branch -M main", shell=True)
    subprocess.run(f"git commit -m 'First Commit'", shell=True)
    subprocess.run(f"git push -u origin main", shell=True)


def git_push(message):
    subprocess.run(f'git commit -am "{message}"', shell=True)
    subprocess.run(f"git push", shell=True)

def git_pull():
    subprocess.run(f'git commit -am "First commit the work before the pull"', shell=True)
    subprocess.run(f'git pull', shell=True)


def git_reset(cnt):
    subprocess.run(f"git reset main{'^' * cnt}", shell=True)



if __name__ == "__main__":
    inputs = input('push or pull or reset? : ').strip()
    if inputs == 'pull':
        git_pull()
    elif inputs == 'push':
        try:
            message = input('message : ').strip()
            git_push(message)
        except Exception as e:
            git_pull()
            git_push(message)
    elif inputs == 'reset':
        cnt = int(input('number of commits : ').strip())
        git_reset(cnt)