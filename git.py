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
    # subprocess.run(f"git pull --rebase", shell=True)
    with open(os.path.expanduser('~') +'/github_personal_access_token.pkl', 'rb') as f:
        data = pickle.load(f)
        username = data['username']
        password = data['password']
    subprocess.run("git add -A", shell=True)
    subprocess.run(f'git commit -m "{message}"', shell=True)
    child = pexpect.spawn("git push -u origin main")
    child.logfile = sys.stdout.buffer
    child.expect("Username for 'https://github.com':")
    child.sendline(username)
    child.expect(f"Password for 'https://{username}@github.com':")
    child.sendline(password)
    child.expect(pexpect.EOF)


def git_pull():
    with open(os.path.expanduser('~') + '/github_personal_access_token.pkl', 'rb') as f:
        data = pickle.load(f)
        username = data['username']
        password = data['password']
    child = pexpect.spawn("git pull")
    child.logfile = sys.stdout.buffer
    child.expect("Username for 'https://github.com':")
    child.sendline(username)
    child.expect(f"Password for 'https://{username}@github.com':")
    child.sendline(password)
    child.expect(pexpect.EOF)


def git_reset(cnt):
    subprocess.run(f"git reset main{'^' * cnt}", shell=True)



if __name__ == "__main__":
    inputs = input('push or pull or reset? : ')
    if inputs == 'pull':
        git_pull()
    elif inputs == 'push':
        try:
            message = input('message : ')
            git_push(message)
        except Exception as e:
            git_pull()
            git_push(message)
    elif inputs == 'reset':
        cnt = int(input('number of commits : '))
        git_reset(cnt)