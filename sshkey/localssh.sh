#!/bin/sh
# https://qiita.com/HamaTech/items/21bb9761f326c4d4aa65
ssh-keygen -t rsa
mkdir ~/.ssh
mv id_rsa.pub ~/.ssh
mv id_rsa ~/.ssh
cd ~/.ssh # 移動しておく
# .ssh配下にauthorized_keysがない場合
mv id_rsa.pub authorized_keys
chmod 600 authorized_keys && chmod 600 id_rsa
