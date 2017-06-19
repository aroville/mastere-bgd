from time import mktime, strptime
from slackclient import SlackClient

slack_token = 'xoxp-81192550499-81259236482-199090900118-23494688baa78114e263b808753ef48c'

sc = SlackClient(slack_token)


def list_files():
    ts = mktime(strptime('01/05/2017', '%d/%m/%Y'))
    response = sc.api_call('files.list', ts_to=ts)
    if response['total'] == 0:
        return None
    files = response['files']
    return [f['id'] for f in files if f['editable']]


def delete(file):
    # see https://api.slack.com/methods/files.delete for more options
    ans = sc.api_call('files.delete', file=file)
    if ans['ok']:
        global nb_deleted
        nb_deleted += 1
        print('deleted {} files'.format(nb_deleted))
    else:
        print('error: {}'.format(ans['error']))


if __name__ == '__main__':
    files_ids = list_files()
    nb_deleted = 0
    while files_ids is not None:
        for file_id in files_ids:
            delete(file_id)
        files_ids = list_files()
    print('Done')
