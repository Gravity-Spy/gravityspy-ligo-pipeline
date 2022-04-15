import panoptes_client
import datetime

subject = panoptes_client.Subject()
subject.links.project = '9979'
subject.metadata['date'] = datetime.datetime.now().strftime('%Y%m%d')
subject.add_location('subject_file3x4.png')
subject.metadata['Filename1'] = 'subject_file3x4.png'
subject.save()
subjectset = panoptes_client.SubjectSet.find(103434)
subjectset.add(subject)



# fatal error - Internal error: TP_NUM_C_BUFS too small: 50