import telebot
import torch
import logging
from torchvision import transforms
import config
from handler import *
import numpy as np
import telethon
from telethon import events
from telebot import types

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = telebot.TeleBot(config.TOKEN)
# Указываем список классов
classes = ['artmuseum', 'cathedralvirgin', 'cherevichkin', 'cityadministration', 'cranes', 'dancing', 'gorkimonument', 'gorkipark', 'grandpashukar',        
          'greecetemple', 'grigoriiaksinya', 'lampsfera', 'loremuseum', 'mechanicalheart', 'monumentcossacks', 'monumentrostovchanka',
          'monumentstella', 'mummonument', 'nahalenok', 'paramonovmension', 'pokrovskitemple', 'publiclibrary', 'readerofeveningrostov',
          'rzd', 'sholohovmonument', 'sobornialleyway', 'svai', 'tamozna', 'theatredrami', 'undergroundmosaic', 'voroshmost', 'wheel']
# Загружаем модель в формате файла
model = torch.jit.load('sight_recognizer_pretrained_999.pt')
transform = transforms.Compose( 
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# Добавляем словарь с информацией о достопримечательностях
landmark_info_ru = {
    'artrmuseum': 'Ростовский областной музей изобразительных искусств\n\nОткрытие самого музея состоялось в 1938 году, но еще в начале ХХ века в Ростове-на-Дону было создано Ростово-Нахичеванское общество изящных искусств. Там экспонировались произведения художников из разных городов России, которые впоследствии после революции попали в музей. У истоков создания музея стояли художники М.С. Сарьян и А.Д. Силин, писательница М.А. Шагинян. С первых лет существования музейная коллекция пополнялась экспонатами из других знаменитых музеев — Третьяковской галереи, Эрмитажа и т.д. В настоящее время музей занимает два здания: одно из них расположено в историческом центре, на одной из красивейших улиц города – Пушкинской.\nМузей находится по адресу: пр. Чехова, 60; ул. Пушкинская, 115.\nПосмотреть на карте:',
    'cathedralvirgin': 'Собор Рождества Пресвятой Богородицы\n\nВпервые был построен в 1765 году в виде небольшой церкви. Уже к 1771 г. состояние церкви Рождества Богородицы требовало перестройки. Купцы решили построить новую, более просторную церковь, однако этому помешала осуществиться эпидемия чумы. Только в 1778 г. было совершено освящения нового храма, однако простоял он не долго. В марте 1778 г. императрица Екатерина Великая подписала указ о переселении крымских армян. Впоследствии церковь много раз была перестроена. Окончательный проект с чертежами появился в 1845 году, а в 1847году состоялась закладка собора. Храм был возведен в 1860 году, но еще несколько лет происходила внутренняя отделка. В 1880 г. в соборе проводились первые ремонтные работы, а с 1890 по 1892 г. за средства ростовского купечества проводилась роспись собора (за исключением алтарей) масляными красками. На данный момент в церкви активно ведется богослужение, а перед возвышается памятник Димитрий Ростовскому.\nСобор находится по адресу: ул. Станиславского, 58.\nПосмотреть на карте:',
    'cityadministration': 'Здание Администрации города Ростова-на-Дону\n\nСтроение было возведено в 1899 году по проекту архитектора Александра Никаноровича Померанцева. Изначально оно было построено в стиле эклектики, и было украшено яркими элементами. В годы Великой Отечественной войны была разрушена часть здания, а именно крыша и верхний этаж, парапеты, декор фасадов. В 1949 — 1953 годах здание было восстановлено. Декор фасадов выполнили заново по сохранившимся фрагментам. В послевоенное время до развала СССР в здании находился Ростовский областной комитет КПСС. После этого здание было передано администрации города Ростова-на-Дону, были проведены реставрационные работы, в ходе которых воссоздали угловые купола, фигуры античных богинь и геральдические вставки, а также белые фасады. В 2016 году было принято решение вернуть историческому зданию его первоначальный облик.\nЗдание находится по адресу: ул. Большая Садовая, 47.\nПосмотреть на карте:',
    'gorkipark': 'Парк культуры и отдыха имени Максима Горького \n\nИстория этого парка начинается еще с 1813 года. На месте балки с ручьём были сады городского главы Андрея Ященко, распорядился открыть их для всех желающих. Постепенно благоустраивались соседние участки, и территория расширялась: был сформирован верхний парк и нижний, а также закладка аллей, которая сохранилась до наших дней. В 1868 году архитектор Владимир Шкитко создал первый фонтан «мальчик с тазиком. Сейчас там расположена клумба-фонтан со скульптурой «Цапли». В 1932 году в Городском саду (ныне парке) имени М. Горького в октябре 1935 года был установлен прижизненный памятник писателю, уничтоженный в 1942 году при бомбежке во время Великой Отечественной войны. В 1999 году перед входом в парк был вмонтирован в асфальт бронзовый памятный «Центр города». Идея этого знака в том, что и к западу, и к востоку от этого знака до границ города одинаковое расстояние.\nПарк расположен в Ленинском районе города, между Пушкинской, Большой Садовой улицами, переулком Семашко и Будённовским проспектом.\nПосмотреть на карте:',
    'greecetemple': 'Греческая Благовещенская церковь \n\nЦерковь была построена в Ростове-на-Дону в начале XX века. Храм находился в Ткачёвском переулке (ныне Университетский) у его пересечения с Мало-Садовой улицей (ныне улица Суворова). Средства на строительство были выделены греками, проживавшими в городе. Самое большое пожертвование сделал владелец табачной фабрики Ахиллес Асланиди. Строительство храма завершилось в 1909 году. Позже здание храма было передано одной из советских школ. Внутри разместился спортзал, а позже Театр кукол (Университетский переулок, 46). После перестройки г. Ростова-на-Дону стал обсуждаться вопрос о возможном восстановлении. В результате в 2003 году храм был размещен на новое место недалеко от Донской публичной библиотеки. Строить его решили в византийском стиле.\n Храм расположен по адресу: Университетский пер., 58.\nПосмотреть на карте:',
    'lampsfera': 'Скульптурные шары "Пушкинские герои" \n\nШары были установлены в 1980‑е годы. Их автором был Анатолий Скнарин. На аллее только два шара, а планировалось поставить три, но не хватило денег. Первый шар — с сюжетами из жизни Пушкина, второй — из романа «Евгений Онегин», а третий должен был быть с изображением сказок.\nПолюбоваться светящимися шарами можно на ул. Пушкинской.\nПосмотреть на карте:', 
    'loremuseum': 'Ростовский областной музей краеведения \n\nОснован в 1937 году в связи с образованием Ростовской области. Пережив революцию и гражданскую войну, с 1920г. музей возобновил свою деятельность как Донской областной советский музей искусств и древностей, поэтому были выделены средства для приобретения картин, скульптур, предметов антиквариата. Затем в 1926 г. было решено реорганизовать Донской музей и создать Северо-Кавказский музей горских народов. 700 экспонатов были переданы в новый музей. После того как был создан Азово-Черноморский край, Музей горских народов потерял свою актуальность, и его экспонаты частично были вывезены в Ставрополь. Новый музей г. Ростова-на-Дону был создан на основе оставшихся экспонатов. В 1951 г. музею было выделено здание в г. Ростове по ул. Энгельса, 87 (Б.Садовая, 79), где после капитального ремонта и реконструкции 7 ноября 1957 года была открыта новая экспозиция.\nМузей находится по адресу: ул. Большая Садовая., 79\nПосмотреть на карте:',
    'mechanicalheart': 'Скульптура «Индустриальное сердце» \n\nЭтот необычный арт-объект был создан на Набережной г. Ростова-на-Дону в 2014 году. Он символизирует город Ростов-на-Дону как сердце индустрии Юга России. Вечером сердце подсвечивается и включаются шестерёнки, что символизирует дыхание и жизнь.\nПолюбоваться этой необычной скульптурой можно на ул. Береговой.\nПосмотреть на карте:', 
    'monumentcossacks': 'Памятник «Не вернувшимся казакам» \n\nТоржественная церемония открытия монумента прошла 17 сентября 2016 года. Он был создан на средства, пожертвованные на благотворительность. Автор памятника — известный скульптор Анатолий Скнарин. Главная часть монумента — конь без всадника. Рядом стоит казачка с маленьким ребёнком. На мраморном постаменте находится доска со следующей надписью: «Дона седого сынам, жизнь отдавшим за счастье Отечества нашего, слава вечная!».\nПолюбоваться скульптурой можно на пересечении улицы Московской и проспекта Будённовского.\nПосмотреть на карте:',
    'monumentrostovchanka': 'Скульптурная композиция «Ростовчанка» \n\nСкульптурная композиция появилась в 1984 году. Автор монумента — А.А. Скнарин. Он изобразил ростовчанку как юнную девушку с распущенными волосами. Монумент украшен мраморной доской, на которой золотом выбиты стихи о ростовчанках со следующей строкой: «Ростовчанки — люди всесоюзной красоты!».\nПолюбоваться скульптурой можно на набережной г.Ростова-на-Дону (ул. Береговая). Ориентиром может служить ресторан «Причал 25».\nПосмотреть на карте:', 
    'monumentstella': 'Памятник-стела «Воинам-освободителям г. Ростова-на-Дону от немецко-фашистских захватчиков» \n\nТоржественное открытие памятника состоялось 8 мая в1983 году в честь освобождения Ростова от немецко-фашистских захватчиков. Высота монумента поражает – целых 72 метра! Стелу украшает скульптура крылатой богини победы Ники (в советское время утверждалось, что это Родина-мать), а со стороны Театральной площади — макет ордена Отечественной войны I-ой степени. Внизу можно увидеть барельефы на темы: «Фронт», «Тыл», «Мир». В комплекс также входит и звонница с 12 колоколами.\nПолюбоваться скульптурой можно в центре Ростова-на-Дону на Театральной площади.\nПосмотреть на карте:', 
    'paramonovmension': 'Особняк Николая Парамонова \n\nЗдание построено в 1914 году для книгоиздателя Николая Парамонова. В настоящее время в особняке находится Зональная научная библиотека имени Ю. А. Жданова Южного федерального университета.\nОсобняк находится по адресу: Пушкинская улица, 148.\nПосмотреть на карте:', 
    'pokrovskitemple': 'Храм покрова пресвятой Богородицы \n\nЯвляется одним из самых старейших храмов в городе. Он был основан в 1762 году, с этого момента началась его нелегкая история. Здание несколько раз перестраивалось, меняло свой статус и даже горело по халатности сторожа в 1985 году. В результате сгорела колокольня, которая в 1831 была реставрирована, также было повреждено здание самого храма. 10 августа 1897 году было заложено новое каменное здание, а строительство завершилось в 1909 году. 25 июня 2007 года перед храмом в Покровском сквере был открыт памятник императрице Елизавете Петровне.\nХрам находится по адресу: ул. Б. Садовая, 113б\nПосмотреть на карте:',
    'publiclibrary': 'Донская Государственная Публичная Библиотека \n\nЯвляется центральной библиотекой Ростовской области. Была основана в 1886 году, считается старейшим хранилищем книг в числе 5 млн. экземпляров. В течение своей истории прошла много трансформаций. В самом начале она называлась Ростовской публичной библиотекой, далее — Ростовская государственная публичная библиотека им. К. Маркса, книгохранилище им. К. Маркса, Донская публичная библиотека им. К. Маркса и др. И наконец с 1992 года — Донская государственная публичная библиотека. Донская публичная библиотека — один из крупных культурных центров Ростова-на-Дону. Здесь проходят кинофестивали и различные выставки.\nБиблиотека находится по адресу: ул. Пушкинская, 175А.\nПосмотреть на карте:',
    'sobornialleyway': 'Переулок Соборный \n\nПереулок Соборный является одной из центральных улиц города Ростова-на-Дону. Он существует еще с 1811 года. Его первое название – Донской спуск. 25 октября 1920 года Соборный был переименован в переулок Подбельского (в честь первого наркома почт и телеграфа В. Н. Подбельского). В школе № 15, располагавшейся в переулке в доме 26/71, в 1927-1936 гг. учился Александр Солженицын (мемориальная доска). Сейчас здесь располагается Институт экономики и внешнеэкономических связей ЮФУ. В 1990-е годы переулку возвращено название Соборный. Ближе к пересечению с ул. Большой Садовой в переулке находится памятник сантехнику, пострадавший в 2016 году от рук вандалов.\nПереулок находится в центре города, районе Ленинском.\nПосмотреть на карте:',
    'theatredrami': 'Ростовский академический театр драмы имени М. Горького \n\nДатой основания театра принято считать 23 июня 1863 года, хотя строительство было задумано задолго до этого. Местом был выбран пустырь между недавно объединёнными в один город Ростовом и Нахичеванью-на-Дону. В 1930 году был объявлен Всесоюзный открытый конкурс, на который представили 25 проектов, 6 из которых получили премии. В результате театр был построен в стиле гусеничного трактора. Театр открылся спектаклем «Мятеж» по повести Д. А. Фурманова. Известно, что по ходу действия на сцену была выведена целая конница. С этого момента театр активно продолжал работу, даже во времена Великой Отечественной войны.\nТеатр находится по адресу: Театральная площадь, 1.\nПосмотреть на карте:',
    'undergroundmosaic': 'Подземный пешеходный переход с мозаичными сюжетными композициями \n\nДостопримечательность выполнена в нескольких подземных переходах города в 1970-х — 1980-х годах. Первый подземный переход был построен в  1969—1970 годах на Театральной площади, но отделан мозаикой был первым переход на Центральном рынке на пересечении улицы Московской и проспекта Будённовского — в 1972 году. В этом переходе шесть выходов, а каждый коридор украшен уникальным сюжетом. Среди них есть картины с летним отдыхом молодежи в 1970-х годах, на них рядом с парусниками виден старый Ворошиловский мост. Есть сюжеты с донской жизнью.\nПолюбоваться мозаичными сюжетами можно на Ворошиловском проспекте, Ростов-на-Дону.\nПосмотреть на карте:',
    'cherevichkin' : 'Памятник Вите Черевичкину \n\nУстановлен в Ростове-на-Дону в Пионерском парке, который в 1961 году назвали именем пионера-героя. Витя Черевичкин (1925–1941) — ростовский мальчик, убитый во время немецкой оккупации. Расстрелян он был за то, что вопреки приказу об уничтожении почтовых голубей укрыл от немцев домашнюю голубятню. Автор: ростовский скульптор Н. В. Аведиков.\nПолюбоваться скульптурой можно в Детском парке имени Вити Черевичкина.\nПосмотреть на карте: ', 
    'cranes' : 'Арт-объект "Танцующие журавли" \n\nУстановлен на Аллее влюбленных, которая находится на набережной города Ростов-на-Дону. Журавль - почитаемая многими народами, красивая птица. Он является символом долголетия, здоровья и благополучия, а журавлиный танец олицетворяет красоту, гармонию и любовь! Наверное поэтому скульптурная композиция «Журавли» сразу привлекает внимание. Две гордые птицы сделаны из маленьких металлических пластин, что добавляет скульптуре особой неповторимости. Роза у ног птиц и дерево с замками сзади наводит на мысль, что памятник символизирует любовь и верность, как напутствие новобрачным. Полюбоваться арт-объектом можно на Набережной города по адресу: Береговая улица, 27/1.\nПосмотреть на карте: ',
    'dancing' : 'Скульптура "Южный танец" \n\nЭта удивительная работа, созданная местным скульптором Кареном Парсамяном, при поддержке компании «Донской причал», является подарком городу и его жителям. Идея скульптуры «Южный танец» зародилась в воображении художника ещё в 2018 году. И только в 2024 году проект обрёл свою физическую форму благодаря компании «Донской причал», которая верит в важность искусства и его роль в общественной жизни. Автор выразил свою творческую концепцию в своём стиле, с помощью выразительных мазков, лепнины, изящных линий металла, использованного для создания скульптуры. В скульптуре отражается динамика и плавность движений любящих сердец. Фигуры, парень и девушка, выполнены с утонченной детализацией, передают энергетику страсти и гармонии.\nПолюбоваться скульптурой можно по адресу: Береговая ул., 12В, Ростов-на-Дону.\nПосмотреть на карте: ',
    'gorkimonument' : 'Памятник Максиму Горькому \n\nМонумент появился в ноябре 1961 года. Писатель изображён в полный рост, на нём длинное расстегнутое пальто, руки убраны в карманы. Фигура размещена на квадратной подставке и установлена на высокий гранитный пьедестал красного цвета. Авторы монумента — скульптор М. С. Алексеенко и архитектор Я. А. Ребайн.\nПолюбоваться монументом можно по адресу: Набережная г.Рстова-на-Дону, Береговая улица.\nПосмотреть на карте: ', 
    'grandpashukar' : 'Дед Щукарь \n\nСкульптура создана Николаем Можаевым и была установлена в 1982 году, в то же время, что и «Нахалёнок».Прозвище «Дед Щукарь»,  шутника и балагура из романа  Михаила Шолохова «Поднятая целина», все время попадающего в разные истории, помнят все. Персонаж этот невыдуманный. Дед по прозвищу Щукарь на самом деле жил в Гремячем Логу. Настоящее его имя – Тимофей Воробьев. Однако в жизни, как и в романе, практически никто его по имени не называл. Прозвище, которое, как описано в «Поднятой целине», он получил еще будучи ребенком, оставалось с ним до последних дней его жизни. «Дед Щукарь» был частым гостем М.А. Шолохова.\nПолюбоваться скульптурой можно по адресу: Береговая ул., 27/1. \nПосмотреть на карте: ',
    'grigoriiaksinya' : 'Григорий и Аксинья \n\nПамятник был установлен на набережной Ростова-на-Дону в июне 2013 г., автором стал известный местный скульптор Сергей Олешня. Она была создана по мотивам романа-эпопеи Михаила Шолохова «Тихий Дон».\nПосмотреть на скульптуру можно по адресу: Береговая ул., 31/4, Ростов-на-Дону.\nПосмотреть на карте: ',
    'mummonument' : 'Мать и дитя \n\nКомпозиция была установлена в 1979 году. Он был объявлен Международным годом ребенка. Главным элементом композиции является фигура женщины. Она стоит на одном колене. Левой рукой женщина обнимает стоящего рядом ребенка. Правую руку она вытянула вперед. С ладони взлетает голубь – символ мира. Скульптурная композиция водружена на высокий четырехугольный постамент, облицованный гранитной плиткой. На постамент нанесены строки из стихотворения Максима Горького. Автором памятника является Михаил Минкус.\nПолюбоваться скульптурой можно по адресу: Ростов-на-Дону, парк имени 1 Мая.\nПосмотреть на карте: ',
    'nahalenok' : 'Нахаленок \n\nВ 1982 году в Ростове-на-Дону была установлена скульптура «Нахаленка». Это персонаж «Донских рассказов» Михаила Шолохова. Композиция представляет собой мальчишку, сидящего на плетне. Скульптор – Н.В. Можаев, архитектор В.И. Волошин, соавтор Э.М. Можаева.\nСкульптура находится по адресу: Ростов-на-Дону, Береговая улица.\nПосмотреть на карте: ',
    'readerofeveningrostov' : 'Читатель вечернего Ростова \n\nСкульптура появилась в 2004 г. на улице Большая Садовая, около издательства газеты «Вечерний Ростов». Скульптурная композиция отлита в бронзе и изображает молодого, одетого с иголочки мужчину, расположившегося на скамейке. Он вальяжно привалился к спинке, закинул ногу на ногу, одной рукой облокотился на лавочку, в другой держит газету. Памятник читателю положил начало традиции украшать город оригинальными бронзовыми персонажами.\nСкульптура находится по адресу: Большая Садовая ул., 4.\nПосмотреть на карте: ',
    'rzd' : 'Управление Владикавказкой ЖД \n\nУправление Общества Владикавказской железной дороги, со времени возникновения находившееся в Ростове, несколько десятилетий не имело собственного здания, а размещалось в разных доходных домах, в арендуемых помещениях. И только в 1910 году был объявлен конкурс на лучший проект здания, под которое управление общества приобрело земельный участок на территории Нахичевани (тогда – отдельного города) рядом с Александровским садом. По результатам архитектурного конкурса из 27 представленных проектов для дальнейшей разработки выбрали проект архитектора Вальтера. Три проекта получили премии, но выбран был архитектор Н. Вальтер. А.П. Буткову, занимающему должность гражданского инженера технического отдела службы пути, доверено провести дальнейшие работы, связанные с проектированием рабочих чертежей, фасадов, а также организация строительных работ. Строить здание начали в июле 1911 года, в основном, завершили к концу 1913 года. Стиль, выбранный для оформления фасадов - «железнодорожный модерн» с мотивами эклектики.\nЗдание находится по адресу: Театральная площадь, 4.\nПосмотреть на карте: ',
    'sholohovmonument' : 'Памятник Шолохову \n\nПамятник М. А. Шолохову в Ростове-на-Дону расположен на набережной реки Дон. Он был установлен в 2000 году в честь 95-летия писателя. Монумент представляет собой выполненную в полный рост фигуру писателя на высоком пьедестале. Шолохов изображён в простой одежде, с сигаретой в руке. Другую руку писатель держит в кармане брюк. Скульптура отлита из меди и установлена на высокий мраморный пьедестал высотой один метр. К памятнику ведёт гранитная лестница из четырёх ступеней. Автор монумента — украинский скульптор, лауреат Шолоховской премии Николай Можаев.\nМонумент находится по адресу: Береговая улица, 12с1.\nПосмотреть на карте: ',
    'svai' : 'Сваи из лиственницы \n\nВ самом центре прогулочной зоны Ростовской набережной расположен стеклянный саркофаг с остатками свай из лиственниц, которыми еще в 19 веке был укреплен берег реки Дон. Древесина лиственницы имеет очень интересные свойства. В высушенном виде она почти не гниет и не заражается микроорганизмами, а при взаимодействии с водой затвердевает и практически превращается в камень. Обект находится по адресу: Береговая ул., 12, стр. 3.\nПосмотреть на карте: ',
    'tamozna' : 'Скульптура в честь основания темерницкой таможни \n\nПамятник в честь основания Темерницкой таможни установлен на набережной Ростова-на-Дону 15 декабря 2010 года. Он был открыт в честь Дня рождения города и дня подписания Указа об основании таможни. Скульптурная композиция изображает фигуру таможенного офицера, который держит в руке Торговый устав 1731 года и другие инструкции и наставления для организации работы таможни. Авторы памятника: скульпторы Сергей Олешня и Анатолий Дементьев.\nПамятник находится по адресу: Ростов-на-Дону, Береговая улица\nПосмотреть на карте: ', 
    'voroshmost' : 'Ворошиловский мост \n\nСтроился с 1961 по 1965 год. Его проектированием занимались инженер Н. И. Кузнецов и архитектор Ш. А. Клейман. Возведение Ворошиловского моста завершило план послевоенной реконструкции набережной Дона, начатой в 1947 году по инициативе первого секретаря Ростовского обкома ВКП Н. С. Патоличева. Мост был построен по новой для своего времени технологии. Бетонные блоки весом до 30 тонн каждый соединялись не привычными сваркой или заклепками, как это делалось раньше, а клеем. Это позволило не использовать сварные и болтовые соединения. Между п-образными опорами при помощи клея из бустилата были закреплены прямоугольные железобетонные полые блоки, а сквозь них протянуты стальные тросы, на которые и была нанизана вся конструкция. В 2013—2017 годах была осуществлена реконструкция Ворошиловского моста, было произведено его расширение с двух имеющихся полос до шести. Данного показателя достигли посредством строительства двух новых мостов на месте старого, с отдельным пролётным сечением на 3 полосы движения в одном направлении у каждого, внешне похожих на старый мост. В процессе реконструкции мост был оснащён 4 лифтами, по 2 на каждом берегу, а также козырьками светопрозрачной конструкции над пешеходной частью.\nМост находится по адресу: Ворошиловский проспект.\nПосмотреть на карте: ', 
    'wheel' : 'Колесо обозрения "Одно небо" \n\nБыло запущено в августе 2016 года в Парк Революции, одном из самых крупных и красивых в Ростове-на-Дону. Аттракцион получил романтическое название «Одно небо». Высота колеса обозрения в Ростове составляет 65 метров. На сегодняшний день аттракцион является третьим по высоте во всей России. Самым высоким считается колесо обозрения в Сочи (83,5 метра). Почетное второе место занимает аналогичный аттракцион в Москве (73 метра). Полный круг колесо обозрения совершает примерно за 15 минут. Колесо обозрения находится по адресу: Театральная площадь, 3.\nПосмотреть на карте: '
    # Предполагается добавление новых объектов
}

landmark_info_en = {
    'artrmuseum': 'Rostov Regional Museum of Fine Arts\n\nThe museum itself opened in 1938, but at the beginning of the 20th century, the Rostov-Nakhichevan Society of Fine Arts was created in Rostov-on-Don. Works by artists from different cities of Russia were exhibited there, which later ended up in the museum after the revolution. The artists M.S. Saryan and A.D. Silin, and the writer M.A. Shaginyan were at the origins of the museum. From the first years of its existence, the museum collection was replenished with exhibits from other famous museums - the Tretyakov Gallery, the Hermitage, etc. Currently, the museum occupies two buildings: one of them is located in the historical center, on one of the most beautiful streets of the city - Pushkinskaya.\nThe museum is located at the address: Chekhov Ave., 60; st. Pushkinskaya, 115.\n View on the map:',
    'cathedralvirgin': 'Cathedral of the Nativity of the Blessed Virgin Mary\n\nIt was first built in 1765 as a small church. By 1771, the condition of the Church of the Nativity of the Virgin Mary required reconstruction. Merchants decided to build a new, more spacious church, but this was prevented by an epidemic of plague. Only in 1778 was the new church consecrated, but it did not last long. In March 1778, Empress Catherine the Great signed a decree on the resettlement of Crimean Armenians. Subsequently, the church was rebuilt many times. The final project with drawings appeared in 1845, and in 1847 the foundation stone of the cathedral was laid. The church was erected in 1860, but the interior decoration took several more years. In 1880, the first repairs were carried out in the cathedral, and from 1890 to 1892, the cathedral (except for the altars) was painted with oil paints at the expense of the Rostov merchants. At the moment, worship is actively held in the church, and a monument to Dmitry Rostovsky rises in front. \nThe cathedral is located at the address: ul. Stanislavskogo, 58.\n View on the map:', 
    'cityadministration': 'Building of the Administration of the City of Rostov-on-Don\n\nThe building was erected in 1899 according to the design of the architect Alexander Nikanorovich Pomerantsev. Initially, it was built in the eclectic style, and was decorated with bright elements. During the Great Patriotic War, part of the building was destroyed, namely the roof and the upper floor, parapets, and facade decor. In 1949-1953, the building was restored. The facade decor was redone based on the surviving fragments. In the post-war period, before the collapse of the USSR, the Rostov Regional Committee of the CPSU was located in the building. After that, the building was transferred to the Rostov-on-Don city administration, restoration work was carried out, during which the corner domes, figures of ancient goddesses and heraldic inserts, as well as white facades were recreated. In 2016, it was decided to return the historic building to its original appearance. \nThe building is located at: ul. Bolshaya Sadovaya, 47.\n View on the map:', 
    'gorkipark': 'Maxim Gorky Culture and Recreation Park \n\nThe history of this park begins in 1813. On the site of the ravine with the stream there were gardens of the city head Andrei Yashchenko, who ordered them to be opened to everyone. Gradually, the neighboring areas were improved, and the territory expanded: the upper and lower parks were formed, as well as the laying of alleys, which have survived to this day. In 1868, the architect Vladimir Shkitko created the first fountain "boy with a basin". Now there is a flowerbed-fountain with a sculpture of "Heron". In 1932, in the City Garden (now the park) named after M. Gorky in October 1935, a lifetime monument to the writer was erected, destroyed in 1942 during the bombing during the Great Patriotic War. In 1999, a bronze memorial "City Center" was built into the asphalt in front of the entrance to the park. The idea behind this sign is that the city limits are the same distance to the west and east of this sign.\nThe park is located in the Leninsky District of the city, between Pushkinskaya, Bolshaya Sadovaya Streets, Semashko Lane and Budyonnovsky Avenue.\n View on the map:',
    'greecetemple': 'Greek Annunciation Church \n\nThe church was built in Rostov-on-Don at the beginning of the 20th century. The temple was located in Tkachevsky Lane (now Universitetsky) at its intersection with Malo-Sadovaya Street (now Suvorova Street). The funds for the construction were provided by the Greeks living in the city. The largest donation was made by the owner of a tobacco factory, Achilles Aslanidi. The construction of the temple was completed in 1909. Later, the temple building was given to one of the Soviet schools. A gym was located inside, and later a Puppet Theater (Universitetsky Lane, 46). After the reconstruction of Rostov-on-Don, the issue of possible restoration began to be discussed. As a result, in 2003, the temple was moved to a new location near the Donskoy Public Library. It was decided to build it in the Byzantine style.\n The temple is located at the address: Universitetsky Lane, 58.\n View on the map:',
    'lampsfera': 'Sculptural balls «Pushkin`s Heroes» \n\nThe balls were installed in the 1980s. Their author was Anatoly Sknarin. There are only two balls on the alley, and it was planned to put three, but there was not enough money. The first ball - with scenes from Pushkin`s life, the second - from the novel «Eugene Onegin»\n View on the map:',
    'loremuseum': 'Rostov Regional Museum of Local History \n\nFounded in 1937 in connection with the formation of the Rostov Region. Having survived the revolution and the civil war, since 1920 the museum resumed its activities as the Don Regional Soviet Museum of Arts and Antiquities, so funds were allocated for the acquisition of paintings, sculptures, and antiques. Then in 1926 it was decided to reorganize the Don Museum and create the North Caucasian Museum of Mountain Peoples. 700 exhibits were transferred to the new museum. After the Azov-Black Sea region was created, the Museum of Mountain Peoples lost its relevance, and its exhibits were partially taken to Stavropol. The new museum of Rostov-on-Don was created on the basis of the remaining exhibits. In 1951, the museum was allocated a building in Rostov on the street. Engels, 87 (B. Sadovaya, 79), where after major repairs and reconstruction on November 7, 1957 a new exhibition was opened.\nThe museum is located at: st. Bolshaya Sadovaya, 79\n View on the map:',
    'mechanicalheart': 'Sculpture "Industrial Heart" \n\nThis unusual art object was created on the Embankment of Rostov-on-Don in 2014. It symbolizes the city of Rostov-on-Don as the heart of industry in the South of Russia. In the evening, the heart is illuminated and gears are turned on, which symbolizes breathing and life.\nYou can admire this unusual sculpture on Beregovaya Street.\n View on the map:', 
    'monumentcossacks': 'Monument to the "Cossacks Who Never Returned" \n\nThe grand opening ceremony of the monument took place on September 17, 2016. It was created with funds donated to charity. The author of the monument is the famous sculptor Anatoly Sknarin. The main part of the monument is a riderless horse. Next to it stands a Cossack woman with a small child. On the marble pedestal is a plaque with the following inscription: "Eternal glory to the sons of the gray Don, who gave their lives for the happiness of our Fatherland!"\nYou can admire the sculpture at the intersection of Moskovskaya Street and Budyonnovsky Avenue.\n View on the map:',
    'monumentrostovchanka': 'Sculptural composition "Rostovchanka" \n\nThe sculptural composition appeared in 1984. The author of the monument is A.A. Sknarin. He depicted the Rostovchanka as a young girl with loose hair. The monument is decorated with a marble plaque on which verses about Rostovchankas are carved in gold with the following line: "Rostovchankas are people of all-Union beauty!"\nYou can admire the sculpture on the embankment of Rostov-on-Don (Beregovaya Street). The restaurant "Prichal 25" can serve as a landmark.', 
    'monumentstella': 'Monument-stele "To the soldiers who liberated Rostov-on-Don from the Nazi invaders" \n\nThe grand opening of the monument took place on May 8, 1983 in honor of the liberation of Rostov from the Nazi invaders. The height of the monument is amazing - as much as 72 meters! The stele is decorated with a sculpture of the winged goddess of victory Nike (in Soviet times it was claimed that this was the Motherland), and from the side of Theater Square - a model of the Order of the Patriotic War of the 1st degree. Below you can see bas-reliefs on the themes: "Front", "Rear", "Peace". The complex also includes a belfry with 12 bells. \nYou can admire the sculpture in the center of Rostov-on-Don on Theater Square. \n View on the map:', 
    'paramonovmension': 'Nikolai Paramonov`s Mansion \n\nThe building was built in 1914 for the book publisher Nikolai Paramonov. Currently, the mansion houses the Yu. A. Zhdanov Zonal Scientific Library of the Southern Federal University.\nThe mansion is located at the address: Pushkinskaya Street, 148.', 
    'pokrovskitemple': 'Church of the Intercession of the Holy Virgin \n\nIs one of the oldest churches in the city. It was founded in 1762, and from that moment its difficult history began. The building was rebuilt several times, changed its status, and even burned down due to the caretaker`s negligence in 1985. As a result, the bell tower, which was restored in 1831, burned down, and the building of the church itself was also damaged. On August 10, 1897, a new stone building was laid, and construction was completed in 1909. On June 25, 2007, a monument to Empress Elizabeth Petrovna was unveiled in front of the church in Pokrovsky Square.\nThe church is located at: ul. B. Sadovaya, 113b \n View on the map:',
    'publiclibrary': 'Don State Public Library \n\nIs the central library of the Rostov region. It was founded in 1886, is considered the oldest book depository with 5 million copies. During its history, it has undergone many transformations. At the very beginning, it was called the Rostov Public Library, then the Rostov State Public Library named after K. Marx, the book depository named after K. Marx, the Don Public Library named after K. Marx, etc. And finally, since 1992, the Don State Public Library. The Don Public Library is one of the major cultural centers of Rostov-on-Don. Film festivals and various exhibitions are held here.\nThe library is located at the address: Pushkinskaya St., 175A.\n View on the map:',
    'sobornialleyway': 'Soborny Lane \n\nSoborny Lane is one of the central streets of Rostov-on-Don. It has existed since 1811. Its first name was Donskoy Spusk. On October 25, 1920, Soborny was renamed Podbelsky Lane (in honor of the first People`s Commissar of Posts and Telegraphs V.N. Podbelsky). Alexander Solzhenitsyn studied at School No. 15, located in the lane at 26/71, in 1927-1936 (memorial plaque). Now the Institute of Economics and Foreign Economic Relations of SFedU is located here. In the 1990s, the lane was given back its name Soborny. Closer to the intersection with Bolshaya Sadovaya Street in the lane there is a monument to a plumber, which was damaged by vandals in 2016.\nThe lane is located in the city center, in the Leninsky district.\n View on the map:',
    'theatredrami': 'Rostov Academic Drama Theatre named after M. Gorky \n\nThe date of the theatre`s foundation is considered to be June 23, 1863, although the construction was conceived long before that. The site chosen was a vacant lot between the recently united cities of Rostov and Nakhichevan-on-Don. In 1930, an All-Union open competition was announced, to which 25 projects were submitted, 6 of which received prizes. As a result, the theatre was built in the style of a caterpillar tractor. The theatre opened with the play "Mutiny" based on the story by D.A. Furmanov. It is known that during the action, an entire cavalry was brought onto the stage. From that moment on, the theatre actively continued to work, even during the Great Patriotic War.\nThe theatre is located at the address: Theatre Square, 1.\n View on the map:',
    'undergroundmosaic': 'Underground pedestrian crossing with mosaic plot compositions \n\nThe landmark was created in several underground crossings of the city in the 1970s and 1980s. The first underground crossing was built in 1969-1970 on Teatralnaya Square, but the first crossing to be decorated with mosaics was the one on the Central Market at the intersection of Moskovskaya Street and Budyonnovsky Avenue — in 1972. This crossing has six exits, and each corridor is decorated with a unique plot. Among them are paintings of young people on a summer vacation in the 1970s, with the old Voroshilovsky Bridge visible next to sailboats. There are scenes with life on the Don.\nYou can admire the mosaic scenes on Voroshilovsky Avenue, Rostov-on-Don.\n View on the map:',
    'cherevichkin': 'The monument to Vita Cherevichkin \n\nWas erected in Rostov-on-Don in Pioneer Park, which was named after the pioneer hero in 1961. Vitya Cherevichkin (1925-1941) was a Rostov boy who was killed during the German occupation. He was shot because, contrary to the order for the destruction of carrier pigeons, he hid a domestic dovecote from the Germans. Author: Rostov sculptor N. V. Avedikov.\nYou can admire the sculpture in the Children`s Park named after Vitya Cherevichkin.\nView on the map: ',
    'cranes': 'The Dancing Cranes art object is located on Lovers` Alley, which is located on the embankment of Rostov-on-Don. The crane is a beautiful bird revered by many peoples. It is a symbol of longevity, health and well-being, and the crane dance embodies beauty, harmony and love! This is probably why the sculptural composition "Cranes" immediately attracts attention. The two proud birds are made of small metal plates, which adds a special uniqueness to the sculpture. A rose at the feet of the birds and a tree with locks at the back suggest that the monument symbolizes love and fidelity, as a farewell to the newlyweds.\nYou can admire the art object on the city`s Embankment at 27/1 Beregovaya Street.\nView on the map: ',
    'dancing' : 'The sculpture "Southern Dance" is an amazing work created by local sculptor Karen Parsamyan, with the support of the Donskoy Pier company, and is a gift to the city and its residents. The idea of the sculpture "Southern Dance" originated in the artist`s imagination back in 2018. It was only in 2024 that the project found its physical form thanks to the Donskoy Pier company, which believes in the importance of art and its role in public life. The author expressed his creative concept in his own style, using expressive brushstrokes, stucco, and graceful lines of metal used to create the sculpture. The sculpture reflects the dynamics and smoothness of the movements of loving hearts. The figures, a boy and a girl, are made with exquisite detail, conveying the energy of passion and harmony.\nYou can admire the sculpture at the address: Beregovaya St., 12B, Rostov-on-Don.\nView on the map: ',
    'gorkimonument' : 'Monument to Maxim Gorky \n\nThe monument appeared in November 1961. The writer is depicted in full height, wearing a long unbuttoned coat, with his hands in his pockets. The figure is placed on a square stand and mounted on a high red granite pedestal. The authors of the monument are sculptor M. S. Alekseenko and architect Ya. A. Rebain.\nYou can admire the monument at the address: EmbankmentRostov-on-Don, Beregovaya Street.\nView on the map: ',
    'grandpashukar' : 'Grandfather Shchukar\n\nThe sculpture was created by Nikolai Mozhaev and was installed in 1982, at the same time as the Nahalenok.Everyone remembers the nickname "Grandfather Shchukar", the joker and buffoon from Mikhail Sholokhov`s novel "The Raised Virgin Land", who always gets into different stories. This character is not fictional. The grandfather, nicknamed Shchukar, actually lived in Gremyachy Log. His real name is Timofey Vorobyov. However, in real life, as in the novel, almost no one called him by name. The nickname, which, as described in "Raised Virgin Land", he received as a child, remained with him until the last days of his life. "Grandfather Shchukar" was a frequent guest of M.A. Sholokhov.\nYou can admire the sculpture at the address: Beregovaya St., 27/1.\nView on the map: ',
    'grigoriiaksinya' : 'Grigory and Aksinya \n\nThe memorial was installed on the embankment of Rostov-on-Don in June 2013, the author was the famous local sculptor Sergey Oleshnya. It was created based on Mikhail Sholokhov`s epic novel The Quiet Don.\nYou can view the sculpture at the address: Beregovaya St., 31/4, Rostov-on-Don.\nView on the map: ',
    'mummonument' : 'Mother and Child \n\nThe composition was established in 1979. It was declared the International Year of the Child. The main element of the composition is the figure of a woman. She`s on one knee. A woman hugs a child standing next to her with her left hand. She held out her right hand. A pigeon flies up from the palm of your hand, a symbol of peace. The sculptural composition is mounted on a high quadrangular pedestal lined with granite tiles. The lines from Maxim Gorky`s poem are applied to the pedestal. The author of the monument is Mikhail Minkus.\nYou can admire the sculpture at the address: Rostov-on-Don, Park named after 1 May.\nView on the map: ',
    'nahalenok' : 'Nahalenok \n\n In 1982, the sculpture was installed in Rostov-on-Don. This is a character in Mikhail Sholokhov`s "Don Stories." The composition represents a boy sitting on a fence. The sculptor is N.V. Mozhaev, the architect is V.I. Voloshin, the co–author is E.M. Mozhaev.\nThe sculpture is located at Rostov-on-Don, Beregovaya Street.\nView on the map: ',
    'readerofeveningrostov' : 'Reader of the evening Rostov \n\nThe sculpture appeared in 2004 on Bolshaya Sadovaya Street, near the publishing house of the newspaper "Evening Rostov". The sculptural composition is cast in bronze and depicts a young, smartly dressed man sitting on a bench. He leaned back impressively, crossed his legs, leaned on the bench with one hand, and held a newspaper in the other. The monument to the reader marked the beginning of the tradition of decorating the city with original bronze characters.\nThe sculpture is located at 4 Bolshaya Sadovaya Street.\nView on the map: ',
    'rzd' : 'Vladikavkaz Railway Administration \n\nThe Administration of the Vladikavkaz Railway Company, located in Rostov since its inception, did not have its own building for several decades, but was located in various apartment buildings and rented premises. It was only in 1910 that a competition was announced for the best design for the building, for which the company`s administration purchased a plot of land in Nakhichevan (then a separate city) next to the Alexander Garden. As a result of the architectural competition, the design by architect Walter was chosen for further development from 27 submitted projects. Three projects received prizes, but architect N. Walter was chosen. A.P. Butkov, who held the position of civil engineer of the technical department of the track service, was entrusted with further work related to the design of working drawings, facades, and the organization of construction work. Construction of the building began in July 1911 and was mostly completed by the end of 1913. The style chosen for the design of the facades is "railway modernism" with eclectic motifs.\nThe building is located at the address: Teatralnaya Square, 4.\nView on the map: ',
    'sholohovmonument' : 'The Sholokhov Monument \n\nThe monument to M. A. Sholokhov in Rostov-on-Don is located on the embankment of the Don River. It was erected in 2000 in honor of the writer`s 95th birthday. The monument is a life-size figure of the writer on a high pedestal. Sholokhov is depicted in simple clothes, with a cigarette in his hand. The writer holds his other hand in his trouser pocket. The sculpture is cast from copper and mounted on a high marble pedestal one meter high. A granite staircase of four steps leads to the monument. The author of the monument is the Ukrainian sculptor, Sholokhov Prize laureate Nikolai Mozhaev.\nThe monument is located at the address: Beregovaya Street, 12s1.\nView on the map: ',
    'svai' : 'Larch piles\n\nIn the very center of the promenade area of ​​the Rostovskaya Embankment there is a glass sarcophagus with the remains of larch piles, which back in the 19th century strengthened the bank of the Don River. Larch wood has very interesting properties. When dried, it almost does not rot and is not infected with microorganisms, and when interacting with water, it hardens and practically turns into stone. The object is located at the following address: Beregovaya St., 12, building 3.\nView on map: ',
    'tamozna' : 'Sculpture in honor of the founding of the Temernitskaya Customs \n\nThe monument in honor of the founding of the Temernitskaya Customs was erected on the embankment of Rostov-on-Don on December 15, 2010. It was opened in honor of the city`s birthday and the day of signing the Decree on the founding of the customs. The sculptural composition depicts the figure of a customs officer holding in his hand the Trade Charter of 1731 and other instructions and guidelines for organizing the work of the customs. The authors of the monument: sculptors Sergey Oleshnya and Anatoly Dementyev.\nThe monument is located at the following address: Rostov-on-Don, Beregovaya St.\nView on map: ',
    'voroshmost' : 'Voroshilovsky Bridge \n\nIt was built from 1961 to 1965. Its design was carried out by engineer N. I. Kuznetsov and architect Sh. A. Kleiman. The construction of the Voroshilovsky Bridge completed the plan for the post-war reconstruction of the Don embankment, which began in 1947 on the initiative of the first secretary of the Rostov regional committee of the All-Union Communist Party of the Soviet Union N. S. Patolichev. The bridge was built using a technology that was new for its time. Concrete blocks weighing up to 30 tons each were connected not by the usual welding or rivets, as was done before, but with glue. This made it possible to avoid using welded and bolted joints. Rectangular reinforced concrete hollow blocks were secured between the U-shaped supports using bustilate glue, and steel cables were pulled through them, on which the entire structure was strung. In 2013-2017, the Voroshilovsky Bridge was reconstructed, expanding it from two existing lanes to six. This figure was achieved by building two new bridges on the site of the old one, with a separate span section for 3 traffic lanes in one direction for each, similar in appearance to the old bridge. During the reconstruction, the bridge was equipped with 4 elevators, 2 on each bank, as well as canopies of a translucent structure above the pedestrian part. \nThe bridge is located at the address: Voroshilovsky Prospekt. \nView on the map: ',
    'wheel' : 'Ferris wheel "One Sky" \n\nIt was launched in August 2016 in Revolution Park, one of the largest and most beautiful in Rostov-on-Don. The attraction received the romantic name "One Sky". The height of the Ferris wheel in Rostov is 65 meters. Today, the attraction is the third tallest in all of Russia. The Ferris wheel in Sochi is considered the tallest (83.5 meters). The honorable second place is occupied by a similar attraction in Moscow (73 meters). The Ferris wheel makes a full circle in about 15 minutes. The Ferris wheel is located at the address: Teatralnaya Square, 3.\nView on the map: '
}

landmark_photos = {
    'artrmuseum': ['https://cdn.culture.ru/images/edbd187b-0d21-5002-98ca-01a795d712d3','https://cdn.culture.ru/images/b610e337-f62b-539d-8810-44e862e3ce75','https://cdn.culture.ru/images/df4fe1a2-23c7-508c-b09e-d55c78dc21de'],
    'cathedralvirgin': ['https://www.donland.ru/upload/uf/648/Sobor_sayt.jpg','https://cdn.culture.ru/images/ee868072-6277-5678-b41d-9266e8b4aa05','https://sobory.ru/pic/06350/06365_20190115_201403.jpg'],
    'cityadministration': ['https://upload.wikimedia.org/wikipedia/commons/6/6f/Rostov_City_Hall_2021.jpg','https://avatars.mds.yandex.net/i?id=ea70ccdab702a479d8a540c0828b4c6c_l-10110781-images-thumbs&n=13','https://s0.rbk.ru/v6_top_pics/media/img/6/75/755875350576756.jpg'],
    'gorkipark': ['https://kartin.papik.pro/uploads/posts/2023-07/1689208146_kartin-papik-pro-p-kartinki-parka-gorkogo-v-rostove-8.jpg','https://avatars.mds.yandex.net/i?id=9752458fd653aaa3211af8b07a715e8b_l-4033961-images-thumbs&n=13','https://www.donnews.ru/netcat_files/mediacontent/2021/05/25/park_gor_kogo_glavnaya.jpg.webp'],
    'greecetemple': ['https://sun9-46.userapi.com/kX_ypIZcvYj46kWzD4wTnMZJkgM2T9btBS58Lw/HJ4QnAgRj_4.jpg','https://rostoveparhia.ru/upload/iblock/0f3/0f35a333f2624c9d6354a84217476b63.jpg','https://static.elitsy.ru/media/src/e8/d6/e8d61422afe64be98b64271a47f70f14.jpg'],
    'lampsfera': ['https://goldcompass.ru/upload/iblock/f29/f29d5b787f8f8b29ee30832bba736890.jpg','https://avatars.mds.yandex.net/i?id=c02f74350fe0b5720612f9e75813d743_l-5086971-images-thumbs&n=13','https://www.dk.ru/system/ckeditor_pictures/000/166/788_content.jpg?1514323707'],
    'loremuseum': ['https://avatars.dzeninfra.ru/get-zen_doc/3930378/pub_5f57977dccc2347a767a627a_5f57d1b622e26e081ade3860/scale_1200','https://cdn.culture.ru/images/2840dba1-c1b9-5956-92fe-418c62b410f2','https://media.kupo.la/thumbor/unsafe/preset/orig/images/2020/6/30/1593506980-cacdff13-989f-4adf-8138-8b1606e1c1d1.jpeg'],
    'mechanicalheart': ['https://avatars.mds.yandex.net/i?id=ee961b603b375061f8531dfeebb87e6d_l-4800766-images-thumbs&n=13','https://sun9-62.userapi.com/impf/c623917/v623917662/6720/2HHDj3HpC-E.jpg?size=1280x988&quality=96&sign=287a56f985ecf13d54a17a60e851e58f&c_uniq_tag=rUfkAsnb6kcv8MFbTXQ0Zehhm_sOTyu6A2Nq4LxpqO0&type=album','https://sneg5.com/wp-content/uploads/DSCF2536_c.jpg'],
    'monumentcossacks': ['https://avatars.mds.yandex.net/i?id=30ba4df405d4d6e705e2cff00e6567e3_l-4032520-images-thumbs&n=13','https://s4.fotokto.ru/photo/full/579/5797558.jpg','https://photo.foto-planeta.com/view/8/8/6/7/rostov-na-donu-886703.jpg'],
    'monumentrostovchanka': ['https://avatars.dzeninfra.ru/get-zen_doc/2453078/pub_601bed58a4b677679fb6645e_601bef996695f61b58f71b4b/scale_1200','https://avatars.mds.yandex.net/i?id=bd354429277dbe4260f1dee7a8a79ff0_l-5220614-images-thumbs&n=13','https://cont.ws/uploads/pic/2022/10/rostovchanka%20%281%29.jpg'], 
    'monumentstella': ['https://storage.myseldon.com/news-pict-cc/CC8494CF64647F50659A0EF4B139D1AA','https://s1.fotokto.ru/photo/full/659/6592445.jpg','https://otvet.imgsmail.ru/download/u_c251b19e69bfdd65eafde044e90292fd_800.jpg'],
    'paramonovmension': ['https://kartarf.ru/images/heritage/1080/8/83620.jpg','https://rostovo-nakhichevansky-pegas.ru/wp-content/uploads/1-osobnjak-n.-e.-paramonova.jpg','https://writervall.ru/wp-content/uploads/2016/03/DSC09624.jpg'], 
    'pokrovskitemple': ['https://yandex.ru/images/search?from=tabbar&img_url=https%3A%2F%2Frostoveparhia.ru%2Fupload%2Fuf%2Fd53%2Fd53c4e83282b6bba8b93ab1ee4d9c8fb.jpg&lr=39&pos=37&rpt=simage&text=%D1%85%D1%80%D0%B0%D0%BC%20%D0%BF%D0%BE%D0%BA%D1%80%D0%BE%D0%B2%D0%B0%20%D0%BF%D1%80%D0%B5%D1%81%D0%B2%D1%8F%D1%82%D0%BE%D0%B9%20%D0%B1%D0%BE%D0%B3%D0%BE%D1%80%D0%BE%D0%B4%D0%B8%D1%86%D1%8B%20%D1%80%D0%BE%D1%81%D1%82%D0%BE%D0%B2-%D0%BD%D0%B0-%D0%B4%D0%BE%D0%BD%D1%83','https://avatars.mds.yandex.net/i?id=a4adbcaa798fa2ec84b82e8756d5077a_l-8496938-images-thumbs&n=13','https://i.ytimg.com/vi/QP6TACin77U/maxresdefault.jpg'],
    'publiclibrary': ['https://i.bigenc.ru/resizer/resize?sign=hMMfeeZk3ODbrob98cU9WA&filename=vault/124396e0f7d6d1a8587ecda57ec6eb0f.webp&width=1024','https://avatars.dzeninfra.ru/get-zen_doc/271828/pub_663170d220437023d84f7c6a_6631758e748c22224836947a/scale_1200','https://avatars.mds.yandex.net/i?id=575df9007ecd90f9fe5ee6fb822d2712_l-5288552-images-thumbs&n=13'],
    'sobornialleyway': ['https://s.101hotelscdn.ru/uploads/image/hotel_image/655148/1803005.jpg','https://www.krestianin.ru/sites/default/files/title_image/2015-12/i9su3hbrfvu.jpg','https://avatars.mds.yandex.net/i?id=70299046acf767d1ccd867cced615dd8_l-4965078-images-thumbs&n=13'],
    'theatredrami': ['https://img-fotki.yandex.ru/get/15549/55277879.24b/0_c8f3e_f469a85a_orig','https://rostov-na-donu.siss.ru/uploads/firm/gallery/271233/321162/medium.jpg?_=2972652900','https://img51994.poehali.tv/img/2024-01-29/fmt_114_24_uogw0kru_y8.jpg'],
    'undergroundmosaic': ['https://ic.pics.livejournal.com/cr2/25595167/461401/461401_original.jpg','https://avatars.mds.yandex.net/i?id=1e65af61af8f26bd573d4c54e7a247b7_l-9065974-images-thumbs&n=13','https://avatars.mds.yandex.net/get-altay/9686455/2a0000018aad5b384cbc7bd910248a0c4dd9/XXL_height'],
    'cherevichkin': ['https://sun9-48.userapi.com/impf/c834303/v834303828/12ebad/mYBVPRnQ8qA.jpg?size=604x343&quality=96&sign=22efeeb5238782c2f10e6e77c8a67480&type=album', 'https://sun9-61.userapi.com/impg/-165K-5MnE8O7hN6KE-ohsYPOlNDHumH-zFSNw/eG5A4zXyb_k.jpg?size=1024x768&quality=96&sign=94787a4aaa439e567a802868145c8732&c_uniq_tag=tBfwUu3w2JJGZJTV8Pc3znHxtvo4gN5M38T1I7Xb2p8&type=album', 'https://sun9-27.userapi.com/impf/c630822/v630822338/13b2c/5xOzeoenrHw.jpg?size=604x336&quality=96&sign=57323d69b496d5b75145456ccb9d27bf&type=album'],
    'cranes': ['https://dynamic-media-cdn.tripadvisor.com/media/photo-o/1b/9c/18/5b/20200711-232814-largejpg.jpg?w=1200&h=1200&s=1', 'https://i0.photo.2gis.com/images/geo/24/3377699734588365_3c81.jpg', 'https://i7.photo.2gis.com/images/geo/24/3377699760986642_e864.jpg'],
    'dancing': ['https://n1s1.hsmedia.ru/7f/07/7b/7f077b35c45fe3719732b02076f91df1/960x640_0x8ZDm2Rd1_7229520977799414982.jpg', 'https://sun9-59.userapi.com/s/v1/ig2/WH101gbA3eCIOcIXyHl7akYgx6oCVm7cuE09CmGNdMtoXyZjop-39rRoBWHKEy3p0iKPOp9fc_c99KhJoP986grM.jpg?quality=95&as=32x43,48x64,72x96,108x144,160x213,240x320,360x480,480x640,540x720,640x853,720x960,960x1280&from=bu&cs=810x1080'], 
    'gorkimonument': ['https://photo.foto-planeta.com/view/6/2/2/2/rostov-na-donu-622270.jpg', 'https://avatars.mds.yandex.net/i?id=64c8822d96747fb950f64e003e166fa2_l-5255595-images-thumbs&n=13', 'https://static.78.ru/images/uploads/1542885050621.jpg'], 
    'grandpashukar': ['https://fanat1k.ru/e107_files/img/0/19169/370641-00.jpg', 'https://avatars.mds.yandex.net/i?id=213c4b8b975ef163fe97bb5ad6df17a7_l-5235840-images-thumbs&n=13', 'https://images.fooby.ru/1/28/33/1366250'],
    'grigoriiaksinya': ['https://avatars.dzeninfra.ru/get-zen_doc/5218824/pub_61095d27689571481b01fb0b_610b7a27c8b57265598fcf3b/scale_1200', 'https://avatars.dzeninfra.ru/get-zen_doc/1945572/pub_61bfb46933a97442c9dc6ca8_61c471b09e007c50e2e43629/scale_1200', 'https://upload.wikimedia.org/wikipedia/ru/e/e3/%D0%A0%D0%9D%D0%94_%D0%93%D1%80%D0%B8%D0%B3%D0%BE%D1%80%D0%B8%D0%B9-%D0%B8-%D0%90%D0%BA%D1%81%D0%B8%D0%BD%D1%8C%D1%8F.jpg'],
    'mummonument': ['https://sun1-16.userapi.com/s/v1/ig2/v2vfY0L73jnf2T1wi_EVqgDkf6m2rmgHk9XlBjqgDF7ooD8LAFHIx6CnzMJkmlYm1EIBBzK-B75iny7_i5XKMLSp.jpg?quality=95&as=32x19,48x28,72x42,108x63,160x94,240x141,360x211,480x281,540x316,640x375,720x422,1024x600&from=bu&u=KmtdGPIA0iFq7iMRLD-cpUFgkjy1-OVX4uVfLQ-WjuI&cs=1024x600', 'https://avatars.mds.yandex.net/get-altay/5587495/2a0000017d9cea25bd4d9138d6e84b259a8c/XXL_height', 'https://s9.travelask.ru/uploads/hint_place/000/004/612/image/f162fbedd96c2940c034aa8c699135bb.jpg'],
    'nahalenok': ['https://avatars.dzeninfra.ru/get-zen_doc/9712324/pub_64956e139bdf3c2c92aa0e8b_649581096b468327af41b06f/scale_1200', 'https://avatars.dzeninfra.ru/get-zen_doc/271828/pub_656ccf2bf6adc66bb9c1b051_656cd1a7e8284b6ff6c535b2/scale_1200', 'https://i.pinimg.com/736x/38/ef/08/38ef0842e39cdf652c264557fea58561.jpg'],
    'readerofeveningrostov': ['https://s4.fotokto.ru/photo/full/447/4471105.jpg', 'https://img-fotki.yandex.ru/get/4527/20839000.3/0_6482f_2f728178_orig', 'https://static.sobaka.ru/images/image/00/92/93/59/_normal.jpg?v=1516018122'],
    'rzd': ['https://lens-club.ru/public/files/gallery/484b5d3beb869b5909ccdacb31c1179a.jpg', 'https://i6.photo.2gis.com/images/geo/24/3377699743457336_6fb2.jpg', 'https://photo.foto-planeta.com/view/6/2/2/1/rostov-na-donu-622132.jpg'],
    'sholohovmonument': ['https://smorodina.com/uploads/image/image/105815/_______-4.jpg', 'https://smorodina.com/uploads/image/image/105815/_______-4.jpg', 'https://donlib.ru/files/2021/0521/240541.jpg'],
    'svai': ['https://upload.wikimedia.org/wikipedia/commons/0/0f/%D0%9D%D0%B0%D0%B1%D0%B5%D1%80%D0%B5%D0%B6%D0%BD%D0%B0%D1%8F_%D0%A0%D0%BE%D1%81%D1%82%D0%BE%D0%B2%D0%B0-%D0%BD%D0%B0-%D0%94%D0%BE%D0%BD%D1%83.jpg'],
    'tamozna': ['https://s2.fotokto.ru/photo/full/628/6286814.jpg', 'https://avatars.dzeninfra.ru/get-zen_doc/224467/pub_5a4ef60e79885ec8f3b67de5_5a4ef613a936f42d74d5c502/scale_1200', 'https://media-cdn.tripadvisor.com/media/photo-s/1b/8c/9e/60/20200711-230434-largejpg.jpg'], 
    'voroshmost': ['https://don24.ru/uploads/2019/03/%D1%82%D1%80%D0%B0%D0%BD%D1%81%D0%BF%D0%BE%D1%80%D1%82/%D0%BC%D0%BE%D1%81%D1%82503.jpg', 'https://avatars.mds.yandex.net/get-pdb/770122/e807d47f-bb93-4cf4-857f-38ee0be8e271/s1200', 'https://img.championat.com/i/54/01/15606254012096956568.jpg'],
    'wheel': ['https://rostov.ruy.ru/upload/iblock/a6f/a6feec62d90ae51f2516a5b83c0d084d.jpg', 'https://sun9-79.userapi.com/impg/S--SeN9wFVYdWNWof3fPHq1yEVbqz7jpK4gzUw/S_oVPxsSaNE.jpg?size=1280x1280&quality=96&sign=6cf04653788272f267c253c63ed4138c&c_uniq_tag=Irsv7E8PPiWeP0MRl_GPSdepJ9dxlOd2FWjjyzFjY1E&type=album', 'https://avatars.mds.yandex.net/i?id=d6672f4f286d69036004056aa2b18b1c_l-5858704-images-thumbs&ref=rim&n=13&w=1080&h=633']
}

# Словарь для хранения счетчиков пользователей
user_counters = {}
# Добавляем координаты для каждой достопримечательности
landmark_coordinates = {
    'artrmuseum': (47.225548, 39.723394),
    'cathedralvirgin': (47.21726, 39.712),
    'cityadministration': (47.221725 , 39.712036),
    'gorkipark': (47.222360, 39.709785),
    'greecetemple': (47.225813, 39.728252),
    'lampsfera': (47.228362, 39.730068),
    'loremuseum': (47.223702, 39.722353),
    'mechanicalheart': (47.214197, 39.717765),
    'monumentcossacks': (47.2171737, 39.7081968),
    'monumentrostovchanka': (47.2137257, 39.7140768),
    'monumentstella': (47.2261810179196, 39.7457232581504),
    'paramonovmension': (47.226891, 39.72532),
    'pokrovskitemple': (47.225798, 39.732131),
    'publiclibrary': (47.2280335, 39.7266230),
    'sobornialleyway': (47.227017, 39.707709),
    'theatredrami': (47.228504, 39.744747),
    'undergroundmosaic': (47.2168350, 39.709675),
    'cherevichkin': (47.227259, 39.749352),
    'cranes': (47.214050, 39.717099),
    'dancing':(47.213812, 39.715383),
    'gorkimonument': (47.214629, 39.719090), 
    'grandpashukar': (47.214146, 39.716617),
    'grigoriiaksinya': (47.215076, 39.720237),
    'mummonument': (47.227697, 39.742440),
    'nahalenok': (47.214063, 39.716030),
    'readerofeveningrostov': (47.217867, 39.697447),
    'rzd': (47.227094, 39.746951),
    'sholohovmonument': (47.213528, 39.712416),
    'svai': (47.213753, 39.714576),
    'tamozna': (47.212806, 39.706381), 
    'voroshmost': (47.214036, 39.722648),
    'wheel': (47.229361, 39.743276)
}
#-----------------------------------------
# Словарь для хранения выбранного языка пользователя
user_languages = {}

# Проверка на id
@bot.message_handler(commands=['id'])
def send_id(message):
    bot.reply_to(message, f"Ваш chat_id: {message.chat.id}")

# Функция для создания клавиатуры выбора языка
def language_keyboard():
    keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    button_ru = telebot.types.KeyboardButton('/ru')
    button_en = telebot.types.KeyboardButton('/en')
    keyboard.add(button_ru, button_en)
    return keyboard

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет! Я бот, который умеет распознавать достопримечательности г.Ростова-на-Дону. '
                                       'Выберите язык:', reply_markup=language_keyboard())

def get_photo(message):
    photo = message.photo[1].file_id
    file_info = bot.get_file(photo)
    file_content = bot.download_file(file_info.file_path)
    return file_content

"""
@bot.message_handler(commands=['start'])
def start_message(message):
    # Приветственный текст
    bot.send_message(message.chat.id, 'Привет! Я бот, который умеет распознавать достопримечательности г.Ростова-на-Дону. Пришли фото сюда, а я определю какой объект находится на твоем фото и расскажу о нем. Выберите язык: /ru для русского или /en для английского.')
"""
@bot.message_handler(commands=['ru'])
def set_language_ru(message):
    user_languages[message.chat.id] = 'ru'
    bot.send_message(message.chat.id, 'Язык установлен на русский.',
reply_markup=telebot.types.ReplyKeyboardRemove())
    
@bot.message_handler(commands=['en'])
def set_language_en(message):
    user_languages[message.chat.id] = 'en'
    bot.send_message(message.chat.id, 'Language set to English.',
reply_markup=telebot.types.ReplyKeyboardRemove())

@bot.message_handler(commands=['info'])    
def help_message(message):    
    # Раздел "помощь"
    language = user_languages.get(message.chat.id, 'ru')  # По умолчанию русский
    if language == 'ru':
        bot.send_message(message.chat.id, 'Отправьте фотографию достопримечательности и бот ее распознает. Помимо названия объекта бот присылает краткое описание и исторические факты, а также местоположение достопримечательности. Данный бот предназначен только для достопримечательностей г.Ростова-на-Дону\nСоздатель бота: @Nastena2011')
    else:
        bot.send_message(message.chat.id, 'Send a photo of a landmark and the bot will recognize it. In addition to the name of the object, the bot sends a short description and historical facts, as well as the location of the landmark. This bot is intended only for landmarks in Rostov-on-Don\nBot creator: @Nastena2011')

@bot.message_handler(content_types=['sticker'])
def sticker_message(message):
  # Ответ на стикер
    language = user_languages.get(message.chat.id, 'ru')  # По умолчанию русский
    if language == 'ru':
        bot.send_message(message.chat.id, 'Вы отправили стикер.')
    else:
        bot.send_message(message.chat.id, 'You have sent a sticker.')

# Получение данных для карт
def get_map_link(coordinates):
    latitude, longitude = coordinates
    return f"https://www.google.com/maps/@{latitude},{longitude},15z"

def process_confirmation(message, image, landmark_name):
    user_response = message.text.strip().lower()  # Удаляем лишние пробелы
    language = user_languages.get(message.chat.id, 'ru')

    if user_response.lower() in ['да', "yes"]:
        bot.send_message(message.chat.id, 'Спасибо за подтверждение!' if language == 'ru' else 'Thank you for your confirmation!')
    elif user_response.lower() in ['нет', "no"]:
        bot.send_message(
            message.chat.id,
            'Пожалуйста, напишите правильное название достопримечательности или нажмите "Я не знаю".' if language == 'ru' else 'Please write the correct name of the landmark or press "I don\'t know".',
            reply_markup=unknown_landmark_keyboard(language)
        )
        bot.register_next_step_handler(message, process_incorrect_data, image)  # Передаем image
    #elif user_response in ['я не знаю', "i don't know"]:
    #    bot.send_message(message.chat.id, 'Спасибо за обратную связь! Данные сохранены.' if language == 'ru' else 'Thank you for your feedback! Data has been saved.')
        # Сохраняем изображение с меткой "unknown"
    #    save_unknown_image(message.chat.id, image)
    else:
        bot.send_message(
            message.chat.id,
            'Вы можете ввести название достопримечательности или нажать "Я не знаю".' if language == 'ru' else 'You can enter the name of the landmark or press "I don\'t know".',
            reply_markup=unknown_landmark_keyboard(language)
        )
        bot.register_next_step_handler(message, process_confirmation, image, landmark_name)


def unknown_landmark_keyboard(language):
    keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    if language == 'ru':
        button = telebot.types.KeyboardButton('Я не знаю')
    else:
        button = telebot.types.KeyboardButton("I don't know")
    keyboard.add(button)
    return keyboard

    
def process_incorrect_data(message, image):
    # Получаем правильное название от пользователя
    correct_landmark = message.text.strip().lower()
    user_id = message.chat.id
    language = user_languages.get(message.chat.id, 'ru')

    # Увеличиваем счетчик для пользователя
    if user_id not in user_counters:
        user_counters[user_id] = 0
    user_counters[user_id] += 1

    if correct_landmark in ['я не знаю', "i don't know"]:
        # Формируем уникальное имя файла
        timestamp = int(time.time())  # Используем временную метку
        image_path = f'incorrect_data/{user_id}_{user_counters[user_id]}_unknown.jpg'

        # Сохраняем изображение
        save_image(image, image_path)
    else:
        # Заменим пробелы на нижние подчеркивания, чтобы избежать проблем с именами файлов
        correct_landmark = correct_landmark.replace(' ', '_')
        # Сохраняем изображение с правильным названием
        image_path = f'incorrect_data/{user_id}_{user_counters[user_id]}_{correct_landmark}.jpg'  # Формируем путь с названием
        save_image(image, image_path)  # Функция для сохранения изображения

        # Сохраняем метку в текстовом файле или базе данных
        with open('incorrect_labels.txt', 'a') as f:
            f.write(f'{image_path},{correct_landmark}\n')

    bot.send_message(message.chat.id,
                     'Спасибо! Данные сохранены.' if language == 'ru'
                     else 'Thank you! Data has been saved.')


def save_image(image, path):
    # Преобразование тензора в изображение
    image = image.squeeze().permute(1, 2, 0).numpy()  # Преобразование тензора в изображение
    image = (image * 255).astype(np.uint8)  # Приведение к диапазону [0, 255]

    # Преобразование из RGB в BGR (OpenCV ожидает BGR порядок каналов)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Применение двустороннего фильтра для уменьшения шума
    # Это также поможет уменьшить цветовые пятна
    image = cv2.bilateralFilter(image, d=5, sigmaColor=35, sigmaSpace=35)

    # Преобразование в цветовое пространство HSV
    # Это разделяет значения яркости и оттенка, что облегчает манипуляции с цветом
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Улучшение контрастности канала яркости (V)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v = clahe.apply(v)

    # Удаление шума цветовых пятен на канале оттенка (H)
    # Это можно сделать с помощью медианного фильтра
    h = cv2.medianBlur(h, 5)

    # Объединение улучшенных каналов для создания улучшенного изображения
    hsv = cv2.merge((h, s, v))

    # Преобразование обратно в BGR
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Сохранение изображения с высоким качеством
    cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


@bot.message_handler(content_types=['photo'])
def repeat_all_messages(message):
    language = user_languages.get(message.chat.id, 'ru')
    try:
        # Получаем содержимое фото
        file_content = get_photo(message)
        
        # Преобразуем байты в изображение
        image = byte2image(file_content)
        image = transform(image)
        
        # Подготавливаем модель к инференсу
        model.eval()
        image = torch.unsqueeze(image, 0)

        # Получаем предсказание
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        landmark_name = classes[int(preds)]

        print(f'Распознанная достопримечательность: {landmark_name}')  # Отладочное сообщение
        logger.info(f'Распознанная достопримечательность: {landmark_name}')  # Логируем распознанное название

        # Проверяем, является ли объект одной из 17 достопримечательностей
        if landmark_name in classes:
            # Получаем язык пользователя
            #language = user_languages.get(message.chat.id, 'ru')  # По умолчанию русский

            # Проверяем, есть ли информация о достопримечательности
            if language == 'ru':
                info = landmark_info_ru.get(landmark_name)
                coordinates = landmark_coordinates.get(landmark_name) # Определяем координаты объектов
                if info is None or coordinates is None:
                    bot.send_message(message.chat.id, 'Извините, я не смог распознать эту достопримечательность.')
                    return
            else:
                info = landmark_info_en.get(landmark_name)
                coordinates = landmark_coordinates.get(landmark_name)
                if info is None or coordinates is None:
                    bot.send_message(message.chat.id, 'Sorry, I could not recognize this landmark.')
                    return
            # Создаем ссылку на карту
            map_link = get_map_link(coordinates)

            # Если информация найдена, отправляем сообщение
            bot.send_message(message.chat.id, text=f'{landmark_name}\n\n{info}\n\n{map_link}')

            # Отправляем группу фотографий
            media_group = []
            if landmark_name in landmark_photos:
                for image_link in landmark_photos[landmark_name]:
                    media_group.append(telebot.types.InputMediaPhoto(image_link))

                try:
                    bot.send_media_group(message.chat.id, media_group)
                except Exception as e:
                    print(f'Ошибка при отправке изображения: {e}')
            time.sleep(2)
            # Запрашиваем подтверждение от пользователя
            if language == 'ru':
                bot.send_message(message.chat.id, "Я ответил на ваш вопрос? (да/нет)")
            else:
                bot.send_message(message.chat.id, "Have I answered your question? (yes/no)")
            bot.register_next_step_handler(message, process_confirmation, image, landmark_name)

        else:
            # Объект не является одной из 17 достопримечательностей
            language = user_languages.get(message.chat.id, 'ru')  # По умолчанию русский
            if language == 'ru':
                bot.send_message(message.chat.id, 'Извините, я не распознаю этот объект.')
            else:
                bot.send_message(message.chat.id, 'Sorry, I do not recognize this object.')

    except Exception as e:
        logger.error(f'Ошибка: {e}', exc_info=True)  # Логируем ошибку с трассировкой
        language = user_languages.get(message.chat.id, 'ru')  # По умолчанию русский
        if language == 'ru':
            bot.send_message(message.chat.id, 'Упс, что-то пошло не так :(\nОбратитесь в службу поддержки!')
        else:
            bot.send_message(message.chat.id, 'Oops, something went wrong :(\nPlease contact support!')

if __name__ == '__main__':
    import time
    import os
    if not os.path.exists('incorrect_data'):
        os.makedirs('incorrect_data')
    while True:
        try:
            bot.polling(none_stop=True)
        except Exception as e:
            time.sleep(15)
            print('Restart!')
bot.polling(none_stop = True)