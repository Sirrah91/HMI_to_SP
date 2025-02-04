; use .r and copy the code to the SSW IDL terminal
; .r
; otherwise type .r b2bprt.pro (or .run)

; Define the parent directory path
parent_directory = '/nfsscratch/david/NN/data/SDO_HMI_stat/'

; Find all subfolders
subfolders = FILE_SEARCH(parent_directory + '*')

; Iterate over each subfolder file
FOR isub = 0, N_ELEMENTS(subfolders) - 1 do begin

    ; Define the folder path
    folder_path = subfolders[isub] + '/'

    ; Find all files with the "*.field.fits" ending
    field_files = FILE_SEARCH(folder_path + '*[.]field[.]fits')

    ; Iterate over each field file
    FOR i = 0, N_ELEMENTS(field_files) - 1 do begin
        ; Extract the base name without the extension
        base_name = FILE_BASENAME(field_files[i])

        ; Split the base name by dots to extract the prefix
        prefix = STRSPLIT(base_name, '.', /EXTRACT)

        ; Construct the corresponding file array
        prefix[3] = 'inclination'
        incl_file = folder_path + STRJOIN(prefix, '.')
        prefix[3] = 'azimuth'
        azimuth_file = folder_path + STRJOIN(prefix, '.')
        prefix[3] = 'disambig'
        disambig_file = folder_path + STRJOIN(prefix, '.')
        files = [field_files[i], incl_file, azimuth_file, disambig_file]

        read_sdo, files[2], index, azi, /uncomp_delete
        rotated_indices = WHERE(STRPOS(index.history, "rotated") NE -1, count_azi)

        read_sdo, files[3], index, ambig, /uncomp_delete
        rotated_indices = WHERE(STRPOS(index.history, "rotated") NE -1, count_ambig)

        IF (count_azi EQ count_ambig) THEN BEGIN
            hmi_disambig, azi, ambig, 1
        ENDIF ELSE BEGIN
            ; Print error and skip iteration
            print, 'Incompatible files: ', files[2:3]
            GOTO, continue_loop
        ENDELSE

        read_sdo, files[0:2], index, data, /uncomp_delete
        data[*,*,2] = azi

        IF (N_ELEMENTS(index) GT 0) AND (N_ELEMENTS(index.history) GT 0) THEN BEGIN
            ; Find occurrences of "rotated" in index.history
            rotated_indices = WHERE(STRPOS(index.history, "rotated") NE -1, count)

            ; If "rotated" is found in all elements, add 180 to azi
            IF (count EQ 0) THEN BEGIN
                ; Do nothing (pass)
            ENDIF ELSE IF (count EQ 3) THEN BEGIN
                data[*,*,2] += 180  ; to compensate the frame rotation done by im_patch
            ENDIF ELSE BEGIN
                ; Print error and skip iteration
                print, 'Incompatible files: ', files[0:2]
                GOTO, continue_loop
            ENDELSE
        ENDIF

        hmi_b2ptr, index[0], data, bptr, lonlat=lonlat

        prefix[3] = 'ptr'
        SAVE, bptr, lonlat, filename=folder_path + STRJOIN(prefix[0:3], '.') + '.sav'

        continue_loop:  ; Label for skipping to the next iteration
    ENDFOR
ENDFOR
end
