# https://qiita.com/blueray777/items/78eb5983997e4c94e590

$ ssh-add /Users/[user_name]/.ssh/[key-name]
Enter passphrase for /Users/[user_name]/.ssh/[key-name]: <- SSHのパスフレーズを入力
Identity added: /Users/[user_name]/.ssh/[key-name] (<作成時に入力したコメント>)

# check whether  your passward register
$ ssh-add -l

# add ~/.ssh/config at MacOs
# common
Host *
 UseKeychain yes
 AddKeysToAgent yes
