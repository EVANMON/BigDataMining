import json
import numpy as np
import matplotlib as plt
import datetime, time
import pytz
import re
import nltk
import csv

#============================ List of cities in WA and MA ====================================
WA_loc_list =['washington', 'aberdeen', 'airway heights', 'algona', 'anacortes',
              'asotin', 'bainbridge island', 'battle ground', 'bellevue',
              'benton city', 'bingen', 'black diamond', 'blaine', 'bonney lake', 'bothell', 'bremerton',
              'bridgeport', 'brier', 'buckley', 'burien', 'camas', 'carnation',
              'cashmere', 'castle rock', 'centralia', 'chehalis', 'chelan', 'cheney', 'chewelah', 'clarkston',
              'cle elum', 'clyde hill', 'colfax', 'college place', 'colville', 'connell', 'cosmopolis',
              'covington', 'davenport', 'dayton', 'deer park', 'des moines', 'dupont', 'duvall', 'east wenatchee',
              'edgewood', 'edmonds', 'electric city', 'ellensburg', 'elma', 'entiat', 'enumclaw', 'ephrata',
              'everson', 'federal way', 'ferndale', 'fife', 'fircrest', 'forks', 'george', 'gig harbor',
              'gold bar', 'goldendale', 'grand coulee', 'grandview', 'granger', 'granite falls', 'harrington',
              'hoquiam', 'ilwaco', 'issaquah', 'kahlotus', 'kalama', 'kelso', 'kenmore', 'kennewick', 'kent',
              'kettle falls', 'kirkland', 'kittitas', 'la center', 'lacey', 'lake forest park', 'lake stevens',
              'lakewood', 'langley', 'leavenworth', 'liberty lake', 'long beach', 'longview', 'lynden', 'lynnwood',
              'mabton', 'maple valley', 'marysville', 'mattawa', 'mccleary', 'medical lake', 'medina', 'mercer island',
              'mesa', 'mill creek', 'millwood', 'montesano', 'morton', 'moses lake', 'mossyrock',
              'mount vernon', 'mountlake terrace', 'moxee', 'mukilteo', 'napavine', 'newcastle', 'newport',
              'nooksack', 'normandy park', 'north bend', 'north bonneville', 'oak harbor', 'oakville', 'ocean shores',
              'okanogan', 'olympia', 'omak', 'oroville', 'orting', 'othello', 'pacific', 'palouse', 'pasco',
              'pateros', 'pomeroy', 'port angeles', 'port orchard', 'port townsend', 'poulsbo', 'prescott',
              'prosser', 'pullman', 'puyallup', 'rainier', 'raymond', 'redmond', 'renton', 'republic',
              'richland', 'ridgefield', 'ritzville', 'rock island', 'roslyn', 'roy', 'royal city', 'ruston',
              'sammamish', 'seatac', 'seattle', 'sedro-woolley', 'selah', 'sequim', 'shelton', 'shoreline',
              'snohomish', 'snohomish', 'snoqualmie', 'soap lake', 'south bend', 'spangle', 'spokane',
              'spokane valley', 'sprague', 'stanwood', 'stevenson', 'sultan', 'sumas', 'sumner', 'sunnyside',
              'tacoma', 'tekoa', 'tenino', 'tieton', 'toledo', 'tonasket', 'toppenish', 'tukwila', 'tumwater',
              'union gap', 'university place', 'vader', 'vancouver', 'waitsburg', 'walla walla', 'wapato',
              'warden', 'washougal', 'wenatchee', 'west richland', 'white salmon', 'winlock',
              'woodinville', 'woodland', 'woodway', 'yakima', 'yelm', 'zillah']

MA_loc_list = ['massachusetts', 'abington', 'acton', 'acushnet', 'adams', 'agawam', 'alford', 'amesbury',
               'amherst', 'andover', 'aquinnah', 'ashburnham', 'ashby', 'ashfield', 'ashland', 'athol',
               'attleboro', 'avon', 'ayer', 'barnstable', 'barre', 'becket', 'bedford', 'belchertown',
               'belmont', 'berkley', 'berlin', 'bernardston', 'beverly', 'billerica', 'blackstone',
               'blandford', 'bolton', 'boston', 'bourne', 'boxborough', 'boxford', 'boylston', 'braintree',
               'bridgewater', 'brimfield', 'brockton', 'brookfield', 'brookline', 'buckland', 'cambridge',
               'canton', 'carlisle', 'carver', 'charlemont', 'charlton', 'chatham', 'chelmsford', 'chelsea',
               'cheshire', 'chester', 'chesterfield', 'chicopee', 'chilmark', 'clarksburg', 'clinton',
               'cohasset', 'colrain', 'concord', 'conway', 'cummington', 'dalton', 'danvers', 'dartmouth',
               'dedham', 'deerfield', 'dennis', 'dighton', 'douglas', 'dover', 'dracut', 'dudley', 'dunstable',
               'duxbury', 'east bridgewater', 'east brookfield', 'east longmeadow', 'eastham', 'easthampton',
               'easton', 'edgartown', 'egremont', 'erving', 'essex', 'fairhaven', 'fall river', 'falmouth',
               'fitchburg', 'florida', 'foxborough', 'framingham', 'franklin', 'freetown', 'gardner',
               'georgetown', 'gill', 'gloucester', 'goshen', 'gosnold', 'grafton', 'granby', 'granville',
               'great barrington', 'greenfield', 'groton', 'groveland', 'hadley', 'halifax', 'hamilton',
               'hampden', 'hancock', 'hanover', 'hanson', 'hardwick', 'harvard', 'harwich', 'hatfield',
               'haverhill', 'hawley', 'heath', 'hingham', 'hinsdale', 'holbrook', 'holden', 'holland',
               'holliston', 'holyoke', 'hopedale', 'hopkinton', 'hubbardston', 'hudson', 'hull', 'huntington',
               'ipswich', 'kingston', 'lakeville', 'lancaster', 'lanesborough', 'lawrence', 'lee', 'leicester',
               'lenox', 'leominster', 'leverett', 'lexington', 'leygen', 'lincoln', 'littleton',
               'longmeadow', 'lowell', 'ludlow', 'lunernburg', 'lynn', 'lynnfield', 'malden',
               'manchester', 'mansfield', 'marblehead', 'marion', 'marlborough', 'marshfield', 'mashpee',
               'mattapoisett', 'maynard', 'medfield', 'medford', 'medway', 'melrose', 'mendon', 'merrimac',
               'methuen', 'middleborough', 'middlefield', 'middleton', 'milford', 'millbury', 'millis',
               'millville', 'monson', 'montague', 'monterey', 'montgomery', 'mount washington', 'nahant',
               'nantucket', 'natick', 'needham', 'new ashford', 'new bedford', 'new braintree',
               'new marlborough', 'new salem', 'newbury', 'newburyport', 'newton', 'norfolk', 'north adams',
               'north andover', 'north brookfield', 'north reading', 'northampton', 'northborough',
               'northbridge', 'northfield', 'norton', 'norwell', 'norwood', 'oak bluffs', 'oakham',
               'orange', 'orleans', 'otis', 'oxford', 'palmer', 'paxton', 'peabody', 'pelham', 'pembroke',
               'pepperell', 'peru', 'petersham', 'phillipston', 'pittsfield', 'plainfield', 'plainville',
               'plymouth', 'plympton', 'princeton', 'provincetown', 'randolph', 'raynham', 'reading',
               'rehoboth', 'revere', 'richmond', 'rochester', 'rockland', 'rockport', 'rowe', 'rowley',
               'royalston', 'russell', 'rutland', 'salem', 'salisbury', 'sandisfield', 'sandwich',
               'saugus', 'savoy', 'scituate', 'seekonk', 'sharon', 'sheffield', 'shelburne', 'sherborn',
               'shirley', 'shrewsbury', 'shutesbury', 'somerset', 'somerville', 'south hadley',
               'southampton', 'southborough', 'southbridge', 'southwick', 'spencer', 'springfield',
               'sterling', 'stockbridge', 'stoneham', 'stoughton', 'stow', 'sturbridge', 'sudbury',
               'sunderland', 'sutton', 'swampscott', 'swansea', 'taunton', 'templeton', 'tewksbury',
               'tisbury', 'tolland', 'topsfield', 'townsend', 'truro', 'tyngsborough','tyringham',
               'upton', 'uxbridge', 'wakefield', 'wales', 'walpole', 'waltham', 'ware', 'wareham',
               'warren', 'warwick', 'watertown', 'wayland', 'webster', 'wellesley', 'wellfleet',
               'wendell', 'wenham', 'west boylston', 'west bridgewater', 'west brookfield', 'west newbury',
               'west springfield', 'west stockbridge', 'we st tisbury', 'westborough', 'westfield',
               'westford', 'westhampton', 'westminster', 'weston', 'westwood', 'weymouth', 'whately',
               'whiteman', 'wilbraham', 'williamsburg', 'williamstown', 'wilmington', 'winchendon',
               'winchester', 'windsor', 'winthrop', 'woburn', 'worcester', 'worthington', 'wrentham',
               'yarmouth']

#============================ the cities whose names include space ======================================
spec_city_WA = ['airway heights', 'bainbridge island', 'battle ground', 'benton city', 'black diamond',
                'bonney lake', 'castle rock', 'clyde hill', 'college place', 'deer park', 'des moines',
                'east wenatchee', 'electric city', 'federal way', 'gig harbor', 'gold bar',
                'grand coulee', 'granite falls', 'kettle falls', 'la center', 'lake forest park',
                'lake stevens', 'liberty lake', 'long beach', 'maple valley', 'medical lake',
                'mercer island', 'mill creek', 'moses lake', 'mount vernon', 'mountlake terrace',
                'normandy park', 'north bend', 'north bonneville', 'oak harbor', 'ocean shores',
                'port angeles', 'port orchard', 'port townsend', 'rock island', 'royal city',
                'soap lake', 'south bend', 'spokane valley', 'union gap', 'university place',
                'walla walla', 'west richland', 'white salmon']
spec_city_MA = ['east bridgewater', 'east brookfield', 'east longmeadow', 'fall river',
                'great barrington', 'mount washington', 'new ashford', 'new bedford', 'new braintree',
                'new marlborough', 'new salem', 'north adams', 'north andover', 'north attleborough',
                'north brookfield', 'north reading', 'oak bluffs', 'south hadley', 'west boylston',
                'west bridgewater', 'west brookfield', 'west newbury', 'west springfield',
                'west stockbridge', 'west tisbury']

#========= check if the location belongs to MA/WA ================
# Input: the location string                                     #
# Output: label (int)                                            #
# 1: WA     0: MA     -1: other                                  #
#=================================================================
def check_loc(loc):
    regex = re.compile('[^a-z]')
    loc_low = regex.sub('', loc.lower())
    loc_split = re.findall(r"[\w]+", loc_low)
    if 'wa' in loc_split:
        return 1
    elif 'ma' in loc_split:
        return 0
    if any(city in loc_low for city in spec_city_WA):
        return 1
    if any(city in loc_low for city in spec_city_MA):
        return 0
    for city in WA_loc_list:
        if city in loc_split and 'd.c.' not in loc_low and 'dc' not in loc_low:
            return 1
    for city in MA_loc_list:
        if city in loc_split:
            return 0
    return -1
'''
    if any(city in loc_low for city in spec_city_WA):
        return 1
    if any(city in loc_low for city in spec_city_MA):
        return 0
    split_loc = loc_low.split(', ')
    for name in split_loc:
        if name in WA_loc_list and 'd.c.' not in split_loc and :
            return 1
        elif name in MA_loc_list:
            return 0
    return -1
'''

pst_tz = pytz.timezone('US/Pacific')
file = './tweet_data/tweets_#superbowl.txt'
infile = open(file, encoding='utf-8')
twt_content = []
twt_loc = []
for line in infile:
    twt = json.loads(line)
    usr_loc = check_loc(twt['tweet']['user']['location'])
    if usr_loc == 1 or usr_loc == 0:
        twt_text = twt['tweet']['text'].replace('#', ' ')
        twt_text = re.sub('https?:\/\/.*[\r\n]*', '', twt_text)
        twt_content.append(str(twt_text))
        twt_loc.append(usr_loc)

twt_content = map(lambda s: s.strip(), twt_content)
content_loc = zip(twt_content, twt_loc)
with open('filtered_tweets.csv', 'w') as f:
    writer = csv.writer(f)
    for row in content_loc:
        writer.writerow(row)




