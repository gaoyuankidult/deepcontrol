TEMPLATE = app
CONFIG += console
CONFIG += c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    serial_port.cpp \
    main.cpp \
    random.cpp

include(deployment.pri)
qtcAddDeployment()

LIBS += -lzmq

HEADERS += \
    serial_port.h \
    random.h

